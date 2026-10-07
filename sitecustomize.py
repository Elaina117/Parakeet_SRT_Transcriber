"""v12r12: intentionally minimal CUDA path setup.
The v12r9/v12r11 custom preload code produced a ctypes startup error.
Let the correctly-built ONNX Runtime CUDA-12 wheel load the NVIDIA DLLs.
"""
from pathlib import Path
import os, sys

if os.name == "nt":
    site = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia = site / "nvidia"
    dirs = []
    if nvidia.is_dir():
        for d in sorted(nvidia.glob("*/bin")):
            if d.is_dir():
                dirs.append(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass
    os.environ["PARAKEET_CUDA_DLL_DIRS"] = os.pathsep.join(dirs)
