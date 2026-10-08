from __future__ import annotations

from cuda_setup import prepare_cuda_environment
prepare_cuda_environment(verbose=True)

from contextlib import asynccontextmanager
from pathlib import Path
import threading
import time
import uuid
import subprocess
import os
import sys
import re
import gc
import json
import queue
from urllib.request import Request as UrlRequest, urlopen
# In Colab, importing PyTorch first preloads its CUDA/cuDNN libraries for ORT.
# Keep Windows installs independent of the optional MOSS PyTorch environment.
if os.environ.get('PARAKEET_COLAB') == '1':
    import torch  # noqa: F401
import numpy as np
import onnxruntime as ort
import onnx_asr
import imageio_ffmpeg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
TMP = ROOT / 'tmp'; TMP.mkdir(exist_ok=True)
CHECKPOINTS = TMP / 'checkpoints'; CHECKPOINTS.mkdir(exist_ok=True)
MODELS = ROOT / 'models'; MODELS.mkdir(exist_ok=True)
STATIC = ROOT / 'static'; STATIC.mkdir(exist_ok=True)
MOSS_WORKER = ROOT / 'moss_worker.py'
MOSS_PYTHON = (ROOT / '.moss-venv' / 'Scripts' / 'python.exe') if os.name == 'nt' else (ROOT / '.moss-venv' / 'bin' / 'python')
MOSS_READY = ROOT / '.moss-venv' / '.install-complete'
MOSS_CHUNK_SECONDS = 60
SILERO_VAD_DIR = MODELS / 'silero-vad'
COLAB_MODE = os.environ.get('PARAKEET_COLAB', '').strip() == '1'

CUDA_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

state = {
    'ready': False,
    'loading': False,
    'error': None,
    'model': None,
    'providers': ort.get_available_providers(),
    'gpu': None,
    'vad': None,
}
jobs = {}
audio_export_jobs = {}
file_picker_lock = threading.Lock()
denoiser_model_lock = threading.Lock()
vad_model_lock = threading.Lock()


def checkpoint_file(checkpoint_id):
    if not re.fullmatch(r'[0-9a-f]{12}', str(checkpoint_id or '')):
        raise ValueError('Invalid checkpoint id')
    return CHECKPOINTS / f'{checkpoint_id}.json'


def checkpoint_audio(checkpoint_id, enhanced=False):
    suffix = '.denoised.f32' if enhanced else '.f32'
    return CHECKPOINTS / f'{checkpoint_id}{suffix}'


def save_checkpoint(data):
    path = checkpoint_file(data['id'])
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def load_checkpoint(checkpoint_id):
    path = checkpoint_file(checkpoint_id)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('id') != checkpoint_id or not data.get('audio_ready'):
        raise ValueError('再開可能なチェックポイントではありません')
    audio = checkpoint_audio(checkpoint_id, enhanced=data.get('enhanced_ready', False))
    if not audio.is_file():
        raise ValueError('再開用のPCM音声が見つかりません')
    return data


def parse_media_time(value, *, default=None):
    """Accept seconds or a clock value such as HH:MM:SS or MM:SS."""
    if value is None or str(value).strip() == '':
        return default
    value = str(value).strip().replace(',', '.')
    try:
        if ':' not in value:
            result = float(value)
        else:
            parts = value.split(':')
            if len(parts) not in (2, 3):
                raise ValueError
            nums = [float(part) for part in parts]
            if any(n < 0 for n in nums) or any(n >= 60 for n in nums[1:]):
                raise ValueError
            result = nums[-1] + nums[-2] * 60 + (nums[-3] * 3600 if len(nums) == 3 else 0)
    except (ValueError, TypeError):
        raise ValueError('時間は秒数または HH:MM:SS 形式で入力してください')
    if not np.isfinite(result) or result < 0:
        raise ValueError('時間は0以上で指定してください')
    return result

# Language tokens supported by the multilingual Parakeet v3 model.
LANGUAGES = {'auto', 'ja', 'en', 'zh', 'ko', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ar', 'hi', 'nl', 'tr', 'uk', 'vi', 'th', 'id'}
DENOISER_MODELS = {
    'gtcrn': (
        MODELS / 'gtcrn_simple.onnx',
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx',
        'GTCRN超高速（CPU）',
    ),
    'dpdfnet': (
        MODELS / 'dpdfnet_baseline.onnx',
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx',
        'DPDFNet高品質',
    ),
}


def get_gpu():
    try:
        p = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=8
        )
        return p.stdout.strip().splitlines()[0] if p.returncode == 0 and p.stdout.strip() else None
    except Exception:
        return None


def load_model():
    if state['ready'] or state['loading']:
        return
    state['loading'] = True
    state['error'] = None
    try:
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError(f'CUDAExecutionProviderが利用できません。利用可能: {available}')

        # CUDA is primary. CPU remains available only for graph nodes that CUDA EP cannot execute.
        sess_options = ort.SessionOptions()

        model = onnx_asr.load_model(
            'nemo-parakeet-tdt-0.6b-v3',
            path=str(MODELS / 'parakeet-v3'),
            sess_options=sess_options,
            providers=CUDA_PROVIDERS,
        ).with_timestamps()

        state.update(
            ready=True,
            model=model,
            providers=CUDA_PROVIDERS,
            gpu=get_gpu() or 'NVIDIA GPU',
        )
        # One server-side startup message only. The browser also de-duplicates repeated status polling.
        print('Parakeet model loaded successfully.')
        print(f'Execution provider priority: {CUDA_PROVIDERS}')
        print('CUDAExecutionProviderでParakeet準備完了')
    except Exception as e:
        state['error'] = repr(e)
        print('Parakeet model load FAILED:', repr(e))
    finally:
        state['loading'] = False


def srt_time(sec):
    ms = max(0, int(round(sec * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def _display_cue_text(cue):
    text = str(cue.get('text', '')).strip()
    speaker = str(cue.get('speaker') or '').strip()
    return f'[{speaker}] {text}' if speaker else text


def write_srt(cues, path):
    lines = []
    for i, c in enumerate(cues, 1):
        start = max(0.0, float(c['start']))
        end = max(start + 0.20, float(c['end']))
        lines += [str(i), f'{srt_time(start)} --> {srt_time(end)}', _display_cue_text(c), '']
    path.write_text('\n'.join(lines), encoding='utf-8-sig')


def _build_words(result, offset):
    """Convert SentencePiece/subword token timestamps into word-level units.

    onnx-asr returns one timestamp per emitted token and token strings that have
    already had SentencePiece ▁ converted to a leading space. A word can span
    several tokens (e.g. 'epis' + 'ode'), so cue boundaries must never be made
    between tokens that belong to the same word.
    """
    ts = getattr(result, 'timestamps', None)
    toks = getattr(result, 'tokens', None) or []
    if not ts or not toks:
        return []

    n = min(len(ts), len(toks))
    items = []
    for i in range(n):
        tok = str(toks[i] or '')
        if not tok.strip():
            continue
        # Timestamp is the token's start frame. onnx-asr uses 10 ms frame steps
        # multiplied by the model's temporal subsampling factor.
        items.append((float(ts[i]) + offset, tok))

    if not items:
        return []

    words = []
    current_text = ''
    current_start = None
    current_last_ts = None

    for idx, (start, tok) in enumerate(items):
        # onnx-asr decodes SentencePiece ▁ to a literal leading space.
        starts_new_word = bool(current_text) and tok.startswith(' ')
        if starts_new_word:
            words.append({'start': current_start, 'end': start, 'text': current_text.strip()})
            current_text = tok
            current_start = start
        else:
            if not current_text:
                current_start = start
                current_text = tok
            else:
                current_text += tok
        current_last_ts = start

    if current_text:
        words.append({'start': current_start, 'end': current_last_ts + 0.35, 'text': current_text.strip()})

    return [w for w in words if w['text']]


def _punctuated(text):
    return text.rstrip().endswith(('.', '?', '!', ':', ';', '…'))


MIN_CUE_DURATION = 3.0

def build_cues(result, offset, max_chars=84, max_duration=6.5, min_duration=MIN_CUE_DURATION):
    """Build readable subtitle cues without splitting words or subwords."""
    words = _build_words(result, offset)
    if not words:
        return []

    cues = []
    cur = None
    for word in words:
        if cur is None:
            cur = {
                'start': word['start'],
                'end': word['end'],
                'text': word['text'],
                '_last_word_end': word['end'],
            }
            continue

        candidate = f"{cur['text']} {word['text']}"
        duration = word['end'] - cur['start']
        too_long = len(candidate) > max_chars or duration > max_duration

        if too_long:
            cur['text'] = cur['text'].strip()
            cur['end'] = max(cur['end'], cur['_last_word_end'])
            cues.append(cur)
            cur = {
                'start': word['start'],
                'end': word['end'],
                'text': word['text'],
                '_last_word_end': word['end'],
            }
            continue

        cur['text'] = candidate
        cur['end'] = word['end']
        cur['_last_word_end'] = word['end']

        # Prefer sentence/punctuation boundaries when the cue is already long enough.
        if duration >= 3.0 and _punctuated(cur['text']):
            cur['text'] = cur['text'].strip()
            cues.append(cur)
            cur = None

    if cur is not None:
        cur['text'] = cur['text'].strip()
        cur['end'] = max(cur['end'], cur['_last_word_end'])
        cues.append(cur)

    # Remove internal helper keys and enforce monotonically increasing intervals.
    cleaned = []
    previous_end = 0.0
    for c in cues:
        start = max(float(c['start']), previous_end)
        end = max(start + 0.20, float(c['end']))
        cleaned.append({'start': start, 'end': end, 'text': c['text']})
        previous_end = end
    return cleaned


def ensure_denoiser_model(job, model_kind):
    model_path, model_url, model_label = DENOISER_MODELS[model_kind]
    if model_path.exists() and model_path.stat().st_size > 100_000:
        return model_path
    with denoiser_model_lock:
        if model_path.exists() and model_path.stat().st_size > 100_000:
            return model_path
        partial = model_path.with_suffix('.onnx.part')
        job.update(message=f'{model_label}モデルをダウンロード中…')
        try:
            req = UrlRequest(model_url, headers={'User-Agent': 'Parakeet-SRT-Transcriber'})
            with urlopen(req, timeout=30) as response, partial.open('wb') as output:
                while True:
                    data = response.read(1024 * 1024)
                    if not data:
                        break
                    output.write(data)
            if partial.stat().st_size < 100_000:
                raise RuntimeError('音声ノイズ抑制モデルのダウンロード結果が不完全です。')
            os.replace(partial, model_path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    return model_path


def denoise_audio_file(raw, enhanced, log_path, model_kind, job):
    model_path = ensure_denoiser_model(job, model_kind)
    model_label = DENOISER_MODELS[model_kind][2]
    total_bytes = max(raw.stat().st_size, 1)
    job.update(stage='denoise', progress=5, message=f'声のノイズ抑制中（{model_label}一括処理）…')
    started_at = time.monotonic()
    command = [
        sys.executable, str(ROOT / 'denoise_worker.py'),
        str(raw), str(enhanced), str(model_path), model_kind,
    ]
    try:
        with log_path.open('wb') as error_log:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=error_log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            last_done_bytes = 0
            while process.poll() is None:
                if job.get('cancel'):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError('ユーザーにより中止されました')
                done_bytes = min(enhanced.stat().st_size if enhanced.exists() else 0, total_bytes)
                if done_bytes > last_done_bytes:
                    done_seconds = done_bytes / 4 / 16000
                    total_seconds = total_bytes / 4 / 16000
                    elapsed = max(time.monotonic() - started_at, 1e-6)
                    realtime = done_seconds / elapsed
                    job.update(
                        progress=10 + int(23 * done_bytes / total_bytes),
                        message=f'声のノイズ抑制中… {done_seconds:.1f} / {total_seconds:.1f}秒（{realtime:.1f}× realtime）',
                    )
                    last_done_bytes = done_bytes
                time.sleep(0.5)
            return_code = process.wait()
            error_log.flush()
        if return_code:
            details = log_path.read_text(encoding='utf-8', errors='replace')[-3000:]
            raise RuntimeError(f'{model_label}処理: ' + (details or f'worker exit code {return_code}'))
    finally:
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            # On Windows, a killed native inference worker can release its
            # inherited stderr handle a moment after process.wait() returns.
            time.sleep(0.2)
            log_path.unlink(missing_ok=True)


def release_parakeet_for_moss():
    """Free Parakeet's CUDA session before launching the isolated Torch runtime."""
    state.update(ready=False, loading=False, model=None, error=None)
    gc.collect()


def detect_voice_intervals(audio_path, duration, job, *, presence_only=False):
    """Use the existing Silero VAD to identify 1-minute regions containing speech."""
    progress_label = (
        '発話の有無だけを検出中…' if presence_only else 'MOSSの前に発話区間を検出中…')
    with vad_model_lock:
        vad = state.get('vad')
        if vad is None:
            job.update(stage='vad', progress=32,
                       message=f'{progress_label}（初回はモデルを準備します）')
            vad_session_options = ort.SessionOptions()
            vad_session_options.intra_op_num_threads = 1
            vad_session_options.inter_op_num_threads = 1
            vad = onnx_asr.load_vad(
                'silero',
                path=str(SILERO_VAD_DIR),
                sess_options=vad_session_options,
                providers=['CPUExecutionProvider'],
            )
            state['vad'] = vad

    sample_rate = 16000
    block_samples = MOSS_CHUNK_SECONDS * sample_rate
    total_samples = audio_path.stat().st_size // np.dtype('<f4').itemsize
    total_blocks = max(1, (total_samples + block_samples - 1) // block_samples)
    intervals = []
    processed_samples = 0
    options = {
        'threshold': 0.5,
        'min_speech_duration_ms': 500,
        'max_speech_duration_s': 20,
        'min_silence_duration_ms': 500,
        'speech_pad_ms': 100,
    }

    with audio_path.open('rb') as source:
        for block_index in range(total_blocks):
            samples = np.fromfile(source, dtype='<f4', count=block_samples)
            if not samples.size:
                break
            result = vad.segment_batch(
                np.expand_dims(samples, axis=0),
                np.asarray([samples.size], dtype=np.int64),
                sample_rate,
                **options,
            )
            for start, end in next(iter(result)):
                if end <= start:
                    continue
                intervals.append({
                    'start': block_index * MOSS_CHUNK_SECONDS + start / sample_rate,
                    'end': block_index * MOSS_CHUNK_SECONDS + end / sample_rate,
                })
            processed_samples += samples.size
            job.update(
                progress=32 + int(2 * processed_samples / max(total_samples, 1)),
                message=(f'{progress_label} '
                         f'{min(duration, processed_samples / sample_rate):.0f} / {duration:.0f}秒'),
            )

    return intervals


def run_moss_transcription(audio_path, duration, job, jid, checkpoint, resume_from=0.0,
                           use_vad=True, chunk_minutes=1):
    if not MOSS_PYTHON.is_file() or not MOSS_READY.is_file():
        raise RuntimeError('MOSS用ランタイムがありません。install_moss_windows.bat を実行してMOSSモデル対応を追加してください。')

    output_path = TMP / f'{jid}.moss.json'
    log_path = TMP / f'{jid}.moss.log'
    vad_path = TMP / f'{jid}.moss-vad.json'
    vad_argument = None
    if use_vad:
        try:
            intervals = detect_voice_intervals(audio_path, duration, job)
            vad_path.write_text(json.dumps({
                'speech_intervals': intervals,
                'presence_only': False,
            }), encoding='utf-8')
            speech_blocks = len({
                int(x['start'] // MOSS_CHUNK_SECONDS) for x in intervals
            })
            job.update(progress=34, message=(
                f'MOSS-Transcribe-Diarize の準備完了… 発話を含む区間 {speech_blocks}件を処理します'))
            vad_argument = str(vad_path)
        except Exception as exc:
            vad_path.unlink(missing_ok=True)
            print('Silero VAD unavailable; MOSS will process all audio chunks:', repr(exc))
            job.update(progress=34, message='MOSS-Transcribe-Diarize の準備完了…')
    else:
        job.update(progress=34, message=(
            f'VADを使わず、音声を{int(chunk_minutes)}分ごとに処理します…'))
    chunk_seconds = MOSS_CHUNK_SECONDS if use_vad else int(chunk_minutes) * 60
    sample_rate = 16000
    total_samples = audio_path.stat().st_size // 4
    block_samples = chunk_seconds * sample_rate
    tail_samples = total_samples % block_samples
    regular_chunks = (total_samples + block_samples - 1) // block_samples
    if (regular_chunks > 1 and 0 < tail_samples <= 4 * sample_rate
            and round(resume_from * sample_rate) >= (regular_chunks - 1) * block_samples):
        # The worker now joins a <=4 s final tail to the previous MOSS block.
        # If resuming a checkpoint made before that change, rewind and remove
        # cues from the block which must be regenerated with the tail.
        rewind_samples = (regular_chunks - 2) * block_samples
        resume_from = rewind_samples / sample_rate
        rewind_time = float(checkpoint['start_seconds']) + resume_from
        retained_cues = [cue for cue in checkpoint.get('segments', [])
                         if float(cue.get('end', 0.0)) <= rewind_time]
        checkpoint.update(segments=retained_cues, processed_seconds=resume_from,
                          updated_at=time.time())
        save_checkpoint(checkpoint)
        job.update(text='\n'.join(_display_cue_text(cue) for cue in retained_cues),
                   processed=resume_from,
                   message='末尾の短い音声区間を直前の区間に結合して再処理します…')
    command = [str(MOSS_PYTHON), str(MOSS_WORKER), str(audio_path), str(output_path),
               job.get('language', 'ja'), str(chunk_seconds), str(resume_from)]
    if vad_argument:
        command.append(vad_argument)
    job.update(stage='moss', progress=34, message='MOSS-Transcribe-Diarize モデルを読み込み中…',
               processed=resume_from, duration=duration)
    process = None
    messages = queue.Queue()
    initial_cues = [dict(cue) for cue in checkpoint.get('segments', [])]
    checkpoint_cues = [dict(cue) for cue in initial_cues]
    live_texts = [_display_cue_text(cue) for cue in initial_cues]
    worker_env = os.environ.copy()
    # start_windows.bat adds the Parakeet venv's site-packages to PYTHONPATH
    # for the main server. Do not inherit that into MOSS: it runs in its own
    # venv, and the extra path can shadow compatible packages such as
    # huggingface_hub with the older Parakeet copy.
    worker_env.pop('PYTHONPATH', None)
    worker_env.pop('PYTHONHOME', None)
    worker_env['PYTHONNOUSERSITE'] = '1'
    # Force UTF-8 in the Windows worker regardless of the server's active code
    # page; long transcripts may contain characters that CP932 cannot encode.
    worker_env['PYTHONUTF8'] = '1'
    worker_env['PYTHONIOENCODING'] = 'utf-8:replace'
    # The Xet downloader remained idle on this Windows host. Use Hugging Face's
    # regular HTTP file path by default; users can opt back into Xet in their env.
    worker_env.setdefault('HF_HUB_DISABLE_XET', '1')

    try:
        with log_path.open('wb') as error_log:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=error_log,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1,
                                       env=worker_env,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

            def read_messages():
                for item in process.stdout:
                    messages.put(item)
                messages.put(None)

            reader = threading.Thread(target=read_messages, daemon=True, name=f'moss-progress-{jid}')
            reader.start()
            stream_closed = False
            while process.poll() is None or not stream_closed or not messages.empty():
                if job.get('cancel'):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError('ユーザーにより中止されました')
                try:
                    line = messages.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    stream_closed = True
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = event.get('type')
                if kind == 'stage':
                    label = event.get('message', 'モデルを読み込み中…')
                    job.update(message=f'MOSS-Transcribe-Diarize で文字起こし中… {label}')
                elif kind == 'warning':
                    warning = str(event.get('message', '')).strip()
                    if warning:
                        job.setdefault('warnings', []).append(warning)
                        job.update(message=f'MOSS-Transcribe-Diarize: {warning}')
                elif kind == 'chunk':
                    current = int(event.get('index', 0))
                    total_chunks = max(1, int(event.get('total', 1)))
                    processed = float(event.get('processed', 0.0))
                    progress = 35 + int(53 * min(1.0, processed / max(duration, 1.0)))
                    job.update(progress=progress, processed=processed, duration=duration,
                               message=f'MOSS-Transcribe-Diarize で文字起こし中… 区間 {current}/{total_chunks}')
                    new_cues = []
                    base_offset = float(checkpoint['start_seconds'])
                    for segment in event.get('segments', []):
                        text = str(segment.get('text', '')).strip()
                        if not text:
                            continue
                        cue = {
                            'start': max(0.0, float(segment['start']) + base_offset),
                            'end': max(float(segment['start']) + base_offset + 0.2,
                                       float(segment['end']) + base_offset),
                            'text': text,
                            'speaker': str(segment.get('speaker') or '').strip(),
                        }
                        checkpoint_cues.append(cue)
                        new_cues.append(cue)
                        live_texts.append(_display_cue_text(cue))
                    checkpoint.update(segments=checkpoint_cues, processed_seconds=processed,
                                      stage='transcribing', updated_at=time.time())
                    save_checkpoint(checkpoint)
                    if live_texts:
                        job.update(text='\n'.join(live_texts))
                elif kind == 'download':
                    downloaded = int(event.get('downloaded', 0))
                    total_bytes = max(1, int(event.get('total', 1)))
                    pct = 34 + int(6 * min(1.0, downloaded / total_bytes))
                    job.update(progress=pct, message=(
                        f"MOSS-Transcribe-Diarize モデルをダウンロード中… "
                        f"{downloaded / 1048576:.0f} / {total_bytes / 1048576:.0f} MB"
                    ))
                elif kind == 'tokens':
                    job.update(message=f"MOSS-Transcribe-Diarize で文字起こし中… 区間 {event.get('index', 0)}/{event.get('total', 0)}・{event.get('count', 0)} tokens")

            return_code = process.wait()
            error_log.flush()

        if return_code:
            details = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
            raise RuntimeError('MOSS-Transcribe-Diarize: ' + (details or f'worker exit code {return_code}'))
        if not output_path.is_file():
            raise RuntimeError('MOSSが文字起こし結果を出力しませんでした。')
        result = json.loads(output_path.read_text(encoding='utf-8'))
        new_cues = [
            {'start': max(0.0, float(segment['start']) + float(checkpoint['start_seconds'])),
             'end': max(float(segment['start']) + float(checkpoint['start_seconds']) + 0.2,
                        float(segment['end']) + float(checkpoint['start_seconds'])),
             'text': str(segment['text']).strip(),
             'speaker': str(segment.get('speaker') or '').strip()}
            for segment in result.get('segments', []) if str(segment.get('text', '')).strip()
        ]
        checkpoint_cues = initial_cues + new_cues
        checkpoint.update(segments=checkpoint_cues, processed_seconds=duration,
                          stage='transcribing', updated_at=time.time())
        save_checkpoint(checkpoint)
        return checkpoint_cues
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        output_path.unlink(missing_ok=True)
        vad_path.unlink(missing_ok=True)
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            time.sleep(0.2)
            log_path.unlink(missing_ok=True)


def job_runner(jid, src, language, delete_src=True, denoise=False, denoise_model='gtcrn',
               asr_model='parakeet', use_vad=True, start_seconds=0.0, end_seconds=None,
               moss_chunk_minutes=1, checkpoint=None, resume_from=0.0):
    job = jobs[jid]
    job['language'] = language
    checkpoint_id = checkpoint['id'] if checkpoint else job['checkpoint_id']
    raw = checkpoint_audio(checkpoint_id)
    enhanced = checkpoint_audio(checkpoint_id, enhanced=True)
    denoise_log = TMP / f'{jid}.denoise.log'
    out = TMP / f'{jid}.srt'
    completed = False
    audio_ready = bool(checkpoint and checkpoint.get('audio_ready'))
    try:
        if checkpoint:
            start_seconds = float(checkpoint['start_seconds'])
            end_seconds = checkpoint.get('end_seconds')
            duration = float(checkpoint['duration'])
            resume_from = float(checkpoint.get('processed_seconds', 0.0))
            language = checkpoint['language']
            denoise = bool(checkpoint.get('denoise'))
            denoise_model = checkpoint.get('denoise_model', 'gtcrn')
            asr_model = checkpoint.get('asr_model', 'parakeet')
            use_vad = bool(checkpoint.get('use_vad', True))
            moss_chunk_minutes = int(checkpoint.get('moss_chunk_minutes', 1))
            cues = [dict(cue) for cue in checkpoint.get('segments', [])]
            job.update(text='\n'.join(_display_cue_text(cue) for cue in cues),
                       processed=resume_from, duration=duration)
            audio_ready = True
        else:
            job.update(stage='ffmpeg', progress=5, message='FFmpegで音声を16kHz mono PCMへ変換中…')
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            try:
                probe = subprocess.run([ff, '-hide_banner', '-i', str(src)], capture_output=True, text=True,
                                       encoding='utf-8', errors='replace', timeout=60)
                probe_text = probe.stderr or ''
            except subprocess.TimeoutExpired:
                probe_text = ''
            duration_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', probe_text)
            media_duration = (
                int(duration_match.group(1)) * 3600
                + int(duration_match.group(2)) * 60
                + float(duration_match.group(3))
            ) if duration_match else None
            if media_duration is not None and start_seconds >= media_duration:
                raise RuntimeError('開始位置が動画の長さを超えています。')
            actual_end = end_seconds if end_seconds is not None else media_duration
            if actual_end is not None and media_duration is not None:
                actual_end = min(float(actual_end), media_duration)
            selected_duration = max(0.0, actual_end - start_seconds) if actual_end is not None else None
            if selected_duration is not None and selected_duration <= 0:
                raise RuntimeError('終了位置は開始位置より後にしてください。')

            ffmpeg_log = TMP / f'{jid}.ffmpeg.log'
            command = [ff, '-hide_banner', '-loglevel', 'error', '-nostats', '-stats_period', '0.5',
                       '-progress', 'pipe:1', '-y']
            if start_seconds:
                command += ['-ss', f'{start_seconds:.3f}']
            command += ['-i', str(src)]
            if selected_duration is not None:
                command += ['-t', f'{selected_duration:.3f}']
            command += ['-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', str(raw)]
            started_at = time.monotonic()
            try:
                with ffmpeg_log.open('wb') as error_log:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log, text=True,
                                               encoding='utf-8', errors='replace', bufsize=1,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    for line in process.stdout:
                        if job.get('cancel'):
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise RuntimeError('ユーザーにより中止されました')
                        if line.startswith('out_time_us=') or line.startswith('out_time_ms='):
                            try:
                                converted = int(line.partition('=')[2]) / 1_000_000
                            except ValueError:
                                continue
                            elapsed = time.monotonic() - started_at
                            if selected_duration and selected_duration > 0:
                                progress = 5 + int(5 * min(1.0, converted / selected_duration))
                                label = f'{converted:.1f} / {selected_duration:.1f}秒'
                            else:
                                progress = 5 + min(4, int(elapsed / 5))
                                label = f'{converted:.1f}秒変換済み'
                            job.update(progress=progress, message=f'FFmpegで音声をPCMへ変換中… {label}',
                                       processed=converted, duration=selected_duration)
                    return_code = process.wait()
                    error_log.flush()
                if return_code:
                    details = ffmpeg_log.read_text(encoding='utf-8', errors='replace')[-3000:]
                    raise RuntimeError('FFmpeg: ' + details)
            finally:
                ffmpeg_log.unlink(missing_ok=True)

            total_samples = raw.stat().st_size // 4
            if total_samples <= 0:
                raise RuntimeError('音声トラックが見つからないか、音声を抽出できませんでした。')
            duration = total_samples / 16000
            checkpoint = {
                'id': checkpoint_id,
                'source_name': job['filename'].removesuffix('.srt'),
                'language': language,
                'denoise': denoise,
                'denoise_model': denoise_model,
                'asr_model': asr_model,
                'use_vad': use_vad,
                'moss_chunk_minutes': moss_chunk_minutes,
                'start_seconds': start_seconds,
                'end_seconds': actual_end,
                'duration': duration,
                'processed_seconds': 0.0,
                'segments': [],
                'audio_ready': True,
                'stage': 'ready',
                'created_at': time.time(),
                'updated_at': time.time(),
            }
            save_checkpoint(checkpoint)
            audio_ready = True
            job.update(checkpoint_id=checkpoint_id, checkpoint_available=True,
                       processed=0.0, duration=duration)

        audio_path = enhanced if checkpoint and checkpoint.get('enhanced_ready') else raw
        if denoise:
            if audio_path != enhanced:
                if not raw.is_file():
                    raise RuntimeError('再開に必要なPCM音声が見つかりません。')
                if not enhanced.is_file() or enhanced.stat().st_size != raw.stat().st_size:
                    enhanced.unlink(missing_ok=True)
                    denoise_audio_file(raw, enhanced, denoise_log, denoise_model, job)
                audio_path = enhanced
                checkpoint.update(enhanced_ready=True, stage='transcribing', updated_at=time.time())
                save_checkpoint(checkpoint)
                raw.unlink(missing_ok=True)
            elif not enhanced.is_file():
                enhanced.unlink(missing_ok=True)
                raise RuntimeError('再開に必要なノイズ抑制済み音声が見つかりません。')
        else:
            checkpoint.update(stage='transcribing', updated_at=time.time())
            save_checkpoint(checkpoint)

        total = audio_path.stat().st_size // 4
        if total <= 0:
            raise RuntimeError('音声トラックが見つからないか、音声を抽出できませんでした。')
        duration = total / 16000
        # Small windows limit future look-ahead. Keep this at 4 s for low timestamp latency.
        chunk = 4 * 16000
        cues = [dict(cue) for cue in checkpoint.get('segments', [])]
        done = min(total, int(round(resume_from * 16000)))
        if asr_model == 'moss':
            if done < total:
                release_parakeet_for_moss()
                cues = run_moss_transcription(audio_path, duration, job, jid, checkpoint,
                                              resume_from, use_vad=use_vad,
                                              chunk_minutes=moss_chunk_minutes)
            checkpoint.update(segments=cues, processed_seconds=duration,
                              stage='transcribing', updated_at=time.time())
            save_checkpoint(checkpoint)
            job.update(text='\n'.join(_display_cue_text(c) for c in cues), processed=duration)
        else:
            progress_start = 35 if denoise else 12
            progress_span = 60 if denoise else 80
            job.update(stage='transcribe',
                       message=f'Parakeet v3 で文字起こし再開中… {done/16000:.1f}秒' if done else
                       'Parakeet v3 で文字起こし中…（4秒チャンク）', duration=duration,
                       progress=min(95, progress_start + int(progress_span * done / max(total, 1))),
                       processed=done / 16000)

            with audio_path.open('rb') as f:
                f.seek(done * 4)
                while done < total:
                    if job.get('cancel'):
                        raise RuntimeError('ユーザーにより中止されました')
                    data = np.fromfile(f, dtype=np.float32, count=min(chunk, total - done))
                    if not data.size:
                        break
                    t0 = time.perf_counter()
                    # Parakeet v3 auto-detects among its supported languages;
                    # onnx-asr only applies the language option to Whisper/Canary.
                    r = state['model'].recognize(data, sample_rate=16000)
                    elapsed = time.perf_counter() - t0
                    cues.extend(build_cues(r, start_seconds + done / 16000,
                                           min_duration=MIN_CUE_DURATION))
                    done += data.size
                    checkpoint.update(segments=cues, processed_seconds=done / 16000,
                                      stage='transcribing', updated_at=time.time())
                    save_checkpoint(checkpoint)
                    job.update(text='\n'.join(c['text'] for c in cues), processed=done / 16000)
                    pct = progress_start + int(progress_span * done / total)
                    rtf = (data.size / 16000) / max(elapsed, 1e-6)
                    job.update(
                        progress=min(95, pct),
                        message=f'Parakeet v3 で文字起こし中… {done/16000:.1f} / {duration:.1f}秒（{rtf:.1f}× realtime）',
                        processed=done / 16000,
                        rtf=rtf,
                    )

        # Recombine chunk-boundary fragments, but only when timing is contiguous
        # and the combined cue remains readable. This prevents tiny cues such as
        # "And yeah." / "so." while never crossing a real silence.
        merged = []
        for c in cues:
            c = dict(c)
            if not merged:
                merged.append(c)
                continue

            prev = merged[-1]
            gap = float(c['start']) - float(prev['end'])
            combined_len = len(prev['text'].rstrip()) + 1 + len(c['text'].lstrip())
            prev_duration = float(prev['end']) - float(prev['start'])
            same_speaker = prev.get('speaker') == c.get('speaker')
            can_join = gap <= 0.35 and combined_len <= 84 and same_speaker

            # Join tiny fragments at chunk boundaries even if a punctuation mark
            # ended the first fragment. This is a subtitle-layout decision, not
            # a speech-recognition change.
            if can_join and prev_duration < MIN_CUE_DURATION:
                prev['text'] = (prev['text'].rstrip() + ' ' + c['text'].lstrip()).strip()
                prev['end'] = max(float(prev['end']), float(c['end']))
            elif can_join and not _punctuated(prev['text']) and combined_len <= 84:
                prev['text'] = (prev['text'].rstrip() + ' ' + c['text'].lstrip()).strip()
                prev['end'] = max(float(prev['end']), float(c['end']))
            else:
                merged.append(c)

        # Keep every cue visible for at least three seconds. Overlap with the
        # following cue is allowed when their speech starts are close together.
        for c in merged:
            desired_end = float(c['start']) + MIN_CUE_DURATION
            c['end'] = max(float(c['end']), desired_end)
            if c['end'] < c['start']:
                c['end'] = desired_end

        # Keep starts ordered, but do not shorten or shift cues to avoid overlap.
        previous_start = 0.0
        cleaned = []
        for c in merged:
            start = max(float(c['start']), previous_start)
            end = max(start + MIN_CUE_DURATION, float(c['end']))
            cleaned.append({
                'start': start,
                'end': end,
                'text': c['text'].strip(),
                **({'speaker': c['speaker']} if c.get('speaker') else {}),
            })
            previous_start = start
        merged = cleaned

        job.update(stage='srt', progress=97, message='SRT生成中…')
        write_srt(merged, out)
        job.update(
            stage='done', progress=100, message='完了', segments=len(merged),
            download=f'/api/jobs/{jid}/srt', filename=job['filename'],
            text='\n'.join(_display_cue_text(x) for x in merged)
        )
        completed = True
    except Exception as e:
        job.update(stage='error', progress=0, message=str(e))
        print(f'Job {jid} FAILED:', repr(e))
    finally:
        try:
            denoise_log.unlink(missing_ok=True)
            if delete_src:
                src.unlink(missing_ok=True)
            if completed:
                raw.unlink(missing_ok=True)
                enhanced.unlink(missing_ok=True)
                checkpoint_file(checkpoint_id).unlink(missing_ok=True)
                job['checkpoint_available'] = False
            elif audio_ready and checkpoint:
                checkpoint.update(
                    stage='interrupted' if job.get('cancel') else 'error',
                    last_error=job.get('message', '') if job.get('stage') == 'error' else '',
                    updated_at=time.time(),
                )
                save_checkpoint(checkpoint)
                if not checkpoint.get('enhanced_ready'):
                    enhanced.unlink(missing_ok=True)
                job['checkpoint_available'] = True
            else:
                raw.unlink(missing_ok=True)
                enhanced.unlink(missing_ok=True)
                checkpoint_file(checkpoint_id).unlink(missing_ok=True)
        except Exception as cleanup_error:
            print(f'Checkpoint cleanup failed for {checkpoint_id}: {cleanup_error!r}')
        if asr_model == 'moss':
            start_model_loader()


def start_model_loader():
    threading.Thread(target=load_model, daemon=True, name='parakeet-model-loader').start()


@asynccontextmanager
async def lifespan(app):
    start_model_loader()
    yield


app = FastAPI(title='Parakeet SRT Transcriber v12r25', lifespan=lifespan)


@app.get('/')
def root():
    return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-store, no-cache, must-revalidate'})


@app.get('/api/status')
def status():
    return {k: state[k] for k in ('ready', 'loading', 'error', 'providers', 'gpu')} | {
        'moss_available': MOSS_PYTHON.is_file() and MOSS_READY.is_file(),
        'colab_mode': COLAB_MODE,
    }


@app.post('/api/load')
def load():
    start_model_loader()
    return status()


def start_transcription_job(src, language, *, delete_src, source_name, denoise=False,
                            denoise_model='gtcrn', asr_model='parakeet', use_vad=True,
                            moss_chunk_minutes=1, start_seconds=0.0, end_seconds=None,
                            checkpoint=None):
    jid = uuid.uuid4().hex[:12]
    stem = Path(source_name).stem.strip() or 'subtitle'
    checkpoint_id = checkpoint['id'] if checkpoint else jid
    jobs[jid] = {
        'stage': 'queued', 'progress': 0, 'message': 'キューに登録', 'text': '',
        'filename': f'{stem}.srt', 'checkpoint_id': checkpoint_id,
        'checkpoint_available': bool(checkpoint), 'warnings': [],
    }
    args = (jid, src, language, delete_src, denoise, denoise_model, asr_model)
    kwargs = {
        'start_seconds': start_seconds,
        'end_seconds': end_seconds,
        'checkpoint': checkpoint,
        'resume_from': float(checkpoint.get('processed_seconds', 0.0)) if checkpoint else 0.0,
        'use_vad': use_vad,
        'moss_chunk_minutes': moss_chunk_minutes,
    }
    threading.Thread(
        target=job_runner,
        args=args,
        kwargs=kwargs,
        daemon=True,
        name=f'job-{jid}',
    ).start()
    return jid


def parse_requested_range(start_time, end_time):
    try:
        start_seconds = parse_media_time(start_time, default=0.0)
        end_seconds = parse_media_time(end_time, default=None)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if end_seconds is not None and end_seconds <= start_seconds:
        raise HTTPException(400, '終了位置は開始位置より後にしてください。')
    return start_seconds, end_seconds


@app.post('/api/transcribe')
async def transcribe(request: Request, filename: str | None = None, language: str = 'ja', denoise: bool = False,
                     denoise_model: str = 'gtcrn', asr_model: str = 'parakeet',
                     use_vad: bool = True, moss_chunk_minutes: int = 1,
                     start_time: str = '0', end_time: str = ''):
    if not state['ready']:
        raise HTTPException(503, state['error'] or 'Parakeet is not ready')
    if language not in LANGUAGES:
        raise HTTPException(400, f'未対応の字幕言語です: {language}')
    if denoise_model not in DENOISER_MODELS:
        raise HTTPException(400, f'未対応のノイズ抑制モデルです: {denoise_model}')
    if asr_model not in ('parakeet', 'moss'):
        raise HTTPException(400, f'未対応の文字起こしモデルです: {asr_model}')
    if asr_model == 'moss' and not (MOSS_PYTHON.is_file() and MOSS_READY.is_file()):
        raise HTTPException(503, 'MOSS用ランタイムがありません。install_moss_windows.bat を実行してMOSSモデル対応を追加してください。')
    if asr_model == 'moss' and not use_vad and not 1 <= moss_chunk_minutes <= 60:
        raise HTTPException(400, 'VADをオフにする場合の区間は1〜60分で指定してください。')
    start_seconds, end_seconds = parse_requested_range(start_time, end_time)
    content_type = request.headers.get('content-type', '').lower()
    if content_type.startswith('multipart/form-data'):
        raise HTTPException(
            415,
            '古い画面からのアップロード形式です。画面を Ctrl+F5 で再読み込みしてから、もう一度お試しください。',
        )
    if content_type and not content_type.startswith('application/octet-stream'):
        raise HTTPException(415, f'未対応のアップロード形式です: {content_type}')
    if not filename:
        raise HTTPException(400, 'ファイル名が受信できませんでした。画面を Ctrl+F5 で再読み込みしてから、もう一度お試しください。')
    suffix = Path(filename).suffix[:16]
    src = TMP / f'{uuid.uuid4().hex[:12]}{suffix}'
    received = 0
    try:
        # Stream the raw request body to disk. Multipart UploadFile first spools
        # the whole upload and then requires a second full-size copy here.
        with src.open('wb') as f:
            async for chunk in request.stream():
                if chunk:
                    f.write(chunk)
                    received += len(chunk)
        if received == 0:
            raise HTTPException(400, 'ファイルが空です')
    except Exception:
        src.unlink(missing_ok=True)
        raise
    jid = start_transcription_job(src, language, delete_src=True, source_name=filename,
                                  denoise=denoise, denoise_model=denoise_model, asr_model=asr_model,
                                  use_vad=use_vad, moss_chunk_minutes=moss_chunk_minutes,
                                  start_seconds=start_seconds, end_seconds=end_seconds)
    return {'job_id': jid}


class LocalTranscribeRequest(BaseModel):
    path: str
    language: str = 'ja'
    denoise: bool = False
    denoise_model: str = 'gtcrn'
    asr_model: str = 'parakeet'
    use_vad: bool = True
    moss_chunk_minutes: int = 1
    start_time: str = '0'
    end_time: str = ''


class AudioExportRequest(BaseModel):
    path: str


def _run_audio_export(export_id, src, destination):
    export = audio_export_jobs[export_id]
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        probe = subprocess.run([ff, '-hide_banner', '-i', str(src)], capture_output=True,
                               text=True, encoding='utf-8', errors='replace', timeout=60)
        probe_text = probe.stderr or ''
        duration_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', probe_text)
        duration = (
            int(duration_match.group(1)) * 3600
            + int(duration_match.group(2)) * 60
            + float(duration_match.group(3))
        ) if duration_match else None
        export.update(stage='running', progress=1, message='音声を再エンコードせず分離中…')
        log_path = TMP / f'{export_id}.audio-export.log'
        command = [ff, '-hide_banner', '-loglevel', 'error', '-nostats', '-stats_period', '0.5',
                   '-progress', 'pipe:1', '-y', '-i', str(src), '-map', '0:a:0', '-vn',
                   '-c:a', 'copy', '-f', 'matroska', str(destination)]
        try:
            with log_path.open('wb') as error_log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                for line in process.stdout:
                    if not line.startswith(('out_time_us=', 'out_time_ms=')):
                        continue
                    try:
                        converted = int(line.partition('=')[2]) / 1_000_000
                    except ValueError:
                        continue
                    if duration and duration > 0:
                        progress = 2 + int(96 * min(1.0, converted / duration))
                        label = f'{converted:.0f} / {duration:.0f}秒'
                    else:
                        progress = min(95, int(export.get('progress', 1)) + 1)
                        label = f'{converted:.0f}秒処理済み'
                    export.update(progress=progress, message=f'音声をそのまま分離中… {label}')
                return_code = process.wait()
                error_log.flush()
            if return_code:
                details = log_path.read_text(encoding='utf-8', errors='replace')[-2500:]
                raise RuntimeError(details or f'FFmpegが終了コード {return_code} で停止しました')
        finally:
            log_path.unlink(missing_ok=True)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError('音声を抽出できませんでした。音声トラックがあるか確認してください。')
        export.update(stage='done', progress=100, message='音声の抽出が完了しました。',
                      path=str(destination), size=destination.stat().st_size)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        export.update(stage='error', message=f'音声抽出エラー: {exc}')


@app.post('/api/extract_audio')
def extract_audio(payload: AudioExportRequest):
    if COLAB_MODE:
        raise HTTPException(501, '音声抽出ボタンはWindows版で利用できます。Windows版で音声を抽出してからColabへアップロードしてください。')
    try:
        src = Path(payload.path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(400, '指定したパスのファイルが見つかりません。')
    if not src.is_file():
        raise HTTPException(400, '指定したパスはファイルではありません。')
    try:
        import tkinter as tk
        from tkinter import filedialog

        with file_picker_lock:
            root = tk.Tk()
            root.title('')
            root.geometry('1x1+0+0')
            root.attributes('-alpha', 0.0)
            root.attributes('-topmost', True)
            root.update()
            root.lift()
            root.focus_force()
            try:
                selected = filedialog.asksaveasfilename(
                    parent=root,
                    title='Colabへ転送する音声ファイルの保存先',
                    initialdir=str(src.parent),
                    initialfile=f'{src.stem}.audio.mka',
                    defaultextension='.mka',
                    filetypes=[('MKA音声（再エンコードなし）', '*.mka')],
                )
            finally:
                root.destroy()
    except Exception as exc:
        raise HTTPException(500, f'保存先ダイアログを開けませんでした: {exc}')
    if not selected:
        return {'cancelled': True}
    destination = Path(selected).expanduser().resolve()
    if destination == src:
        raise HTTPException(400, '元の動画ファイルと同じ場所には保存できません。')
    if destination.suffix.lower() != '.mka':
        destination = destination.with_suffix('.mka')
    export_id = uuid.uuid4().hex[:12]
    audio_export_jobs[export_id] = {'stage': 'queued', 'progress': 0, 'message': 'FFmpegを準備中…'}
    threading.Thread(target=_run_audio_export, args=(export_id, src, destination),
                     daemon=True, name=f'audio-export-{export_id}').start()
    return {'job_id': export_id}


@app.get('/api/extract_audio/{export_id}')
def get_audio_export(export_id: str):
    export = audio_export_jobs.get(export_id)
    if export is None:
        raise HTTPException(404, '音声抽出ジョブが見つかりません。')
    return export


@app.post('/api/transcribe_local')
def transcribe_local(payload: LocalTranscribeRequest):
    if not state['ready']:
        raise HTTPException(503, state['error'] or 'Parakeet is not ready')
    if payload.language not in LANGUAGES:
        raise HTTPException(400, f'未対応の字幕言語です: {payload.language}')
    if payload.denoise_model not in DENOISER_MODELS:
        raise HTTPException(400, f'未対応のノイズ抑制モデルです: {payload.denoise_model}')
    if payload.asr_model not in ('parakeet', 'moss'):
        raise HTTPException(400, f'未対応の文字起こしモデルです: {payload.asr_model}')
    if payload.asr_model == 'moss' and not (MOSS_PYTHON.is_file() and MOSS_READY.is_file()):
        raise HTTPException(503, 'MOSS用ランタイムがありません。install_moss_windows.bat を実行してMOSSモデル対応を追加してください。')
    if payload.asr_model == 'moss' and not payload.use_vad and not 1 <= payload.moss_chunk_minutes <= 60:
        raise HTTPException(400, 'VADをオフにする場合の区間は1〜60分で指定してください。')
    start_seconds, end_seconds = parse_requested_range(payload.start_time, payload.end_time)
    try:
        src = Path(payload.path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(400, '指定したパスのファイルが見つかりません。サーバーPC上の絶対パスを確認してください。')
    if not src.is_file():
        raise HTTPException(400, '指定したパスはファイルではありません。')
    jid = start_transcription_job(src, payload.language, delete_src=False, source_name=src.name,
                                  denoise=payload.denoise, denoise_model=payload.denoise_model,
                                  asr_model=payload.asr_model, use_vad=payload.use_vad,
                                  moss_chunk_minutes=payload.moss_chunk_minutes,
                                  start_seconds=start_seconds,
                                  end_seconds=end_seconds)
    return {'job_id': jid}


@app.post('/api/pick_file')
def pick_local_file():
    """Open a native file picker on the Windows desktop running this server."""
    if COLAB_MODE:
        raise HTTPException(501, 'Colabではファイル選択ダイアログを使えません。Google Driveをマウントし、/content/drive/MyDrive/... の動画パスを入力してください。')
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Give the native dialog a real, foreground owner. A withdrawn Tk root
        # can leave the Windows file picker behind the browser with no visible hint.
        with file_picker_lock:
            root = tk.Tk()
            root.title('')
            root.geometry('1x1+0+0')
            root.attributes('-alpha', 0.0)
            root.attributes('-topmost', True)
            root.update()
            root.lift()
            root.focus_force()
            try:
                selected = filedialog.askopenfilename(
                    parent=root,
                    title='文字起こしするメディアファイルを選択',
                    filetypes=[
                        ('動画・音声ファイル', '*.mp4 *.mkv *.mov *.avi *.m4v *.webm *.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma'),
                        ('すべてのファイル', '*.*'),
                    ],
                )
            finally:
                root.destroy()
        return {'path': selected or None}
    except Exception as e:
        raise HTTPException(500, f'ファイル選択ダイアログを開けませんでした: {e}')


@app.get('/api/jobs/{jid}')
def getjob(jid):
    if jid not in jobs:
        raise HTTPException(404, 'job not found')
    return jobs[jid]


@app.get('/api/checkpoints')
def list_checkpoints():
    active_ids = {
        job.get('checkpoint_id') for job in jobs.values()
        if job.get('stage') not in ('done', 'error')
    }
    items = []
    for path in CHECKPOINTS.glob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            checkpoint_id = data.get('id')
            if not data.get('audio_ready') or not checkpoint_audio(checkpoint_id).is_file():
                continue
            items.append({
                'id': checkpoint_id,
                'source_name': data.get('source_name', 'media'),
                'language': data.get('language', 'ja'),
                'asr_model': data.get('asr_model', 'parakeet'),
                'start_seconds': data.get('start_seconds', 0.0),
                'end_seconds': data.get('end_seconds'),
                'duration': data.get('duration', 0.0),
                'processed_seconds': data.get('processed_seconds', 0.0),
                'stage': data.get('stage', 'interrupted'),
                'last_error': data.get('last_error', ''),
                'updated_at': data.get('updated_at', data.get('created_at', 0.0)),
                'active': checkpoint_id in active_ids,
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    items.sort(key=lambda item: item['updated_at'], reverse=True)
    return {'checkpoints': items}


@app.post('/api/checkpoints/{checkpoint_id}/resume')
def resume_checkpoint(checkpoint_id: str):
    if not state['ready']:
        raise HTTPException(503, state['error'] or 'Parakeet is not ready')
    try:
        checkpoint = load_checkpoint(checkpoint_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(404, str(exc))
    if any(job.get('checkpoint_id') == checkpoint_id and job.get('stage') not in ('done', 'error')
           for job in jobs.values()):
        raise HTTPException(409, 'この処理はすでに実行中です。')
    if checkpoint.get('asr_model') == 'moss' and not (MOSS_PYTHON.is_file() and MOSS_READY.is_file()):
        raise HTTPException(503, 'MOSS用ランタイムが見つかりません。')
    if checkpoint.get('asr_model') == 'moss':
        start_seconds = float(checkpoint.get('start_seconds', 0.0))
    else:
        start_seconds = float(checkpoint.get('start_seconds', 0.0))
    jid = start_transcription_job(
        checkpoint_audio(checkpoint_id, enhanced=checkpoint.get('enhanced_ready', False)),
        checkpoint.get('language', 'ja'),
        delete_src=False, source_name=checkpoint.get('source_name', 'subtitle'),
        denoise=checkpoint.get('denoise', False),
        denoise_model=checkpoint.get('denoise_model', 'gtcrn'),
        asr_model=checkpoint.get('asr_model', 'parakeet'),
        use_vad=checkpoint.get('use_vad', True),
        moss_chunk_minutes=checkpoint.get('moss_chunk_minutes', 1),
        start_seconds=start_seconds, end_seconds=checkpoint.get('end_seconds'),
        checkpoint=checkpoint,
    )
    return {'job_id': jid}


@app.delete('/api/checkpoints/{checkpoint_id}')
def delete_checkpoint(checkpoint_id: str):
    try:
        metadata_path = checkpoint_file(checkpoint_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if any(job.get('checkpoint_id') == checkpoint_id and job.get('stage') not in ('done', 'error')
           for job in jobs.values()):
        raise HTTPException(409, '処理中のチェックポイントは削除できません。')
    for path in (metadata_path, checkpoint_audio(checkpoint_id),
                 checkpoint_audio(checkpoint_id, enhanced=True)):
        path.unlink(missing_ok=True)
    return {'ok': True}


@app.post('/api/jobs/{jid}/cancel')
def cancel(jid):
    if jid not in jobs:
        raise HTTPException(404, 'job not found')
    jobs[jid]['cancel'] = True
    return {'ok': True}


@app.get('/api/jobs/{jid}/srt')
def getsrt(jid):
    p = TMP / f'{jid}.srt'
    if not p.exists():
        raise HTTPException(404, 'SRT not ready')
    filename = jobs.get(jid, {}).get('filename', f'{jid}.srt')
    return FileResponse(p, media_type='application/x-subrip', filename=filename)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=os.environ.get('PARAKEET_HOST', '127.0.0.1'), port=5173, reload=False)
