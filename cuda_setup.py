from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_handles = []
_loaded = []


def _site_nvidia_dirs() -> list[str]:
    site = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia = site / "nvidia"
    dirs: list[str] = []
    if nvidia.is_dir():
        for d in sorted(nvidia.glob("*/bin")):
            if d.is_dir():
                dirs.append(str(d))
    return dirs


def _load_dll(path: Path) -> tuple[bool, str]:
    try:
        h = ctypes.WinDLL(str(path))
        _loaded.append((path.name, h))
        return True, "OK"
    except OSError as e:
        return False, repr(e)


def prepare_cuda_environment(verbose: bool = False) -> list[str]:
    """Prepare Windows CUDA wheel DLL lookup and preload core DLLs before ORT import."""
    # Do not allow a legacy system CUDA 11.x install to win over the matching CUDA 11.8
    # runtime bundled in this venv.
    for key in ("CUDA_PATH", "CUDA_HOME", "CUDA_PATH_V12_0", "CUDA_PATH_V11_8"):
        os.environ.pop(key, None)

    if os.name != "nt":
        return []

    dirs = _site_nvidia_dirs()
    # PATH and DLL search path. Keep venv NVIDIA directories first.
    prefix = os.pathsep.join(dirs)
    if prefix:
        old = os.environ.get("PATH", "")
        os.environ["PATH"] = prefix + (os.pathsep + old if old else "")
        for s in dirs:
            try:
                _handles.append(os.add_dll_directory(s))
            except OSError:
                pass

    # First make sure the Microsoft runtime is present/locatable. ORT's CUDA
    # provider is a native Windows DLL and Windows may otherwise report the
    # misleading generic error 126.
    system_root = Path(os.environ.get("SystemRoot", r"C:\\Windows"))
    for name in ("vcruntime140_1.dll", "vcruntime140.dll", "msvcp140.dll"):
        for base in (system_root / "System32", system_root / "SysWOW64"):
            if (base / name).exists():
                _load_dll(base / name)
                break

    # v12r17: ORT 1.18.1 Windows provider in this environment imports
    # CUDA 11.x / cuDNN 8.x DLL names (confirmed by PE import scan).
    # Explicitly preload the exact major-version DLLs required by the provider.
    preferred = [
        ("cublas", "cublasLt64_11.dll"),
        ("cublas", "cublas64_11.dll"),
        ("cuda_runtime", "cudart64_110.dll"),
        ("cuda_nvrtc", "nvrtc64_112_0.dll"),
        ("cufft", "cufft64_10.dll"),
        ("curand", "curand64_10.dll"),
        ("cudnn", "cudnn64_8.dll"),
        ("cudnn", "cudnn_ops_infer64_8.dll"),
        ("cudnn", "cudnn_cnn_infer64_8.dll"),
        ("cudnn", "cudnn_ops_train64_8.dll"),
        ("cudnn", "cudnn_cnn_train64_8.dll"),
    ]
    for package, pattern in preferred:
        base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / package / "bin"
        if not base.is_dir():
            continue
        matches = sorted(base.glob(pattern))
        for p in matches[:3]:
            ok, msg = _load_dll(p)
            if verbose:
                print(f"Preload {p.name}: {msg}")

    return dirs


def loaded_dll_names() -> list[str]:
    return [x[0] for x in _loaded]
