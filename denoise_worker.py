from __future__ import annotations

import sys
import os
import threading
import ctypes
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cuda_setup import prepare_cuda_environment

prepare_cuda_environment(verbose=False)

_thread_state = threading.local()


def physical_core_count() -> int:
    """Return physical CPU cores where the OS exposes processor topology."""
    if os.name == 'nt':
        try:
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            get_info = kernel32.GetLogicalProcessorInformationEx
            get_info.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            get_info.restype = ctypes.c_int
            size = ctypes.c_ulong(0)
            get_info(0, None, ctypes.byref(size))  # RelationProcessorCore
            buffer = ctypes.create_string_buffer(size.value)
            if not get_info(0, buffer, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            offset = 0
            cores = 0
            while offset + 8 <= size.value:
                relationship, record_size = struct.unpack_from('<II', buffer.raw, offset)
                if record_size < 8 or offset + record_size > size.value:
                    break
                if relationship == 0:  # RelationProcessorCore
                    cores += 1
                offset += record_size
            if cores:
                return cores
        except (AttributeError, OSError, ValueError):
            pass

    try:
        import psutil
        count = psutil.cpu_count(logical=False)
        if count:
            return count
    except ImportError:
        pass

    # Prefer an honest fallback over creating workers for virtual cores.
    return max(1, (os.cpu_count() or 1) // 2)


def _denoise_block(samples: np.ndarray, model_path: Path, model_kind: str):
    """Use a separate single-threaded model session in each worker thread."""
    if not hasattr(_thread_state, 'denoiser'):
        import sherpa_onnx

        if model_kind == 'gtcrn':
            model_config = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model_path))
            )
        else:
            model_config = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(model=str(model_path))
            )
        model_config.debug = False
        model_config.num_threads = 1
        # GTCRN is tiny: the GTX 1080's per-inference launch/copy overhead is
        # much higher than its compute cost. CPU inference benchmarks faster
        # here; keep DPDFNet on CUDA because it is substantially heavier.
        provider = 'cpu' if model_kind == 'gtcrn' else 'cuda'
        model_config.provider = provider
        config = sherpa_onnx.OfflineSpeechDenoiserConfig(model=model_config)
        if not config.validate():
            raise RuntimeError(f'Invalid {model_kind.upper()} configuration: {config}')
        _thread_state.denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)

    result = _thread_state.denoiser.run(samples, 16000)
    return np.asarray(result.samples, dtype='<f4')


def main() -> int:
    if len(sys.argv) != 5:
        print('usage: denoise_worker.py INPUT.f32 OUTPUT.f32 MODEL.onnx MODEL_KIND', file=sys.stderr)
        return 2

    source_path, output_path, model_path = map(Path, sys.argv[1:4])
    model_kind = sys.argv[4]
    try:
        if model_kind not in ('gtcrn', 'dpdfnet'):
            raise ValueError(f'Unsupported denoiser model: {model_kind}')
        # Process bounded blocks with the offline API. This sends thousands of
        # frames per inference call instead of invoking the streaming API once
        # per 16 ms frame. Independent blocks can use separate CPU sessions.
        block_samples = 30 * 16000
        workers = physical_core_count() if model_kind == 'gtcrn' else 1
        if model_kind == 'gtcrn':
            print(f'GTCRN CPU workers: {workers} physical cores', file=sys.stderr, flush=True)

        with source_path.open('rb') as source, output_path.open('wb', buffering=0) as output:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='gtcrn-cpu') as pool:
                pending = []
                while True:
                    samples = np.fromfile(source, dtype=np.float32, count=block_samples)
                    if not samples.size:
                        break
                    pending.append(pool.submit(_denoise_block, np.ascontiguousarray(samples), model_path, model_kind))
                    # Bound queued audio to at most twice the worker count and
                    # write completed output in source order.
                    if len(pending) >= workers * 2:
                        pending.pop(0).result().tofile(output)
                        output.flush()
                for future in pending:
                    future.result().tofile(output)
                    output.flush()

        return 0
    except Exception as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
