from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path

from cuda_setup import prepare_cuda_environment, loaded_dll_names

print("Python:", sys.executable)
print("Python version:", sys.version.split()[0])
print("Machine:", platform.machine())
print("CUDA_PATH before fix:", os.environ.get("CUDA_PATH", "<unset>"))

dirs = prepare_cuda_environment(verbose=True)
print("CUDA_PATH after fix:", os.environ.get("CUDA_PATH", "<unset>"))
print("NVIDIA DLL dirs:", dirs)
print("Preloaded DLLs:", loaded_dll_names())

try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.STDOUT,
    ).strip()
    print("GPU:", gpu)
except Exception as e:
    print("GPU query failed:", repr(e))

try:
    ctypes.WinDLL("nvcuda.dll")
    print("nvcuda.dll LoadLibrary: OK")
except OSError as e:
    print("nvcuda.dll LoadLibrary: FAILED", repr(e))
    raise SystemExit(2)

import onnxruntime as ort
print("ORT:", ort.__version__)
print("Providers advertised:", ort.get_available_providers())

capi = Path(ort.__file__).parent / "capi"
provider = capi / "onnxruntime_providers_cuda.dll"
shared = capi / "onnxruntime_providers_shared.dll"
print("CUDA provider DLL:", provider)
print("NOTE: v12r17 intentionally requires CUDA 11.x + cuDNN 8.x for this ORT 1.18.1 provider build.")
print("CUDA provider DLL exists:", provider.is_file())
print("Shared provider DLL exists:", shared.is_file())
for p in (shared, provider):
    if not p.is_file():
        continue
    try:
        ctypes.WinDLL(str(p))
        print(p.name, "LoadLibrary: OK")
    except OSError as e:
        print(p.name, "LoadLibrary: FAILED", repr(e))

# Import table inspection gives actionable names when Windows only returns error 126.
try:
    import pefile
    pe = pefile.PE(str(provider), fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    imports = []
    for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
        try: name = entry.dll.decode('ascii')
        except Exception: name = str(entry.dll)
        imports.append(name)
    print("Provider imported DLLs (expected CUDA 11 / cuDNN 8):")
    for name in imports:
        print("  ", name)
except Exception as e:
    print("PE import scan failed:", repr(e))

try:
    import json
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    # Use an operation that is supported by CUDA EP (MatMul), not Identity.
    # Disable CPU EP graph fallback so a successful session/run cannot silently
    # execute on CPU.
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        "cuda_probe",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 128]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [128, 128])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 128])],
    )
    model = helper.make_model(
        graph,
        producer_name="parakeet-cuda-probe",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    probe = Path(__file__).with_name("cuda_probe.onnx")
    probe.write_bytes(model.SerializeToString())
    print("Probe ONNX IR:", model.ir_version)

    sess_options = ort.SessionOptions()
    sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        sess_options.disable_cpu_ep_fallback = True
    except Exception:
        pass
    sess = ort.InferenceSession(
        str(probe),
        sess_options=sess_options,
        providers=["CUDAExecutionProvider"],
    )
    sess.disable_fallback()
    print("CUDAExecutionProvider session: OK")
    print("Session providers:", sess.get_providers())

    a = np.ones((1, 128), dtype=np.float32)
    b = np.ones((128, 128), dtype=np.float32)
    result = sess.run(None, {"A": a, "B": b})
    print("CUDA probe result[0][0]:", float(result[0][0, 0]))

    # CPU fallback is disabled for this diagnostic session, so completing the
    # CUDA MatMul proves that CUDA executed it without creating an ORT profile.
    if not np.allclose(result[0], 128.0):
        raise RuntimeError("CUDA probe returned an unexpected MatMul result")

    print("CUDA DIAGNOSTIC: PASS")
except Exception as e:
    print("CUDAExecutionProvider session: FAILED")
    print("ERROR:", repr(e))
    raise SystemExit(4)

finally:
    try: probe.unlink()
    except Exception: pass
