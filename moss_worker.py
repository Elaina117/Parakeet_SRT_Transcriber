"""Optional isolated MOSS inference worker for the Parakeet SRT Transcriber."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_url, snapshot_download


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError, ValueError):
        pass


MODEL_ID = 'OpenMOSS-Team/MOSS-Transcribe-Diarize'
SAMPLE_RATE = 16_000
MAX_NEW_TOKENS = max(512, int(os.environ.get('MOSS_MAX_NEW_TOKENS', '5120')))
MOSS_REPETITION_PENALTY = max(1.0, float(os.environ.get('MOSS_REPETITION_PENALTY', '1.08')))
MIN_RETRY_SECONDS = 4
MAX_SPEECH_REGION_SECONDS = 16
SPEECH_REGION_MERGE_GAP_SECONDS = 2.5
SPEECH_REGION_PADDING_SECONDS = 1.25
SILENCE_RMS_THRESHOLD = 1e-5
SILENCE_PEAK_THRESHOLD = 1e-4
LANGUAGES = {
    'auto': None,
    'ja': '日本語', 'en': '英語', 'zh': '中国語', 'ko': '韓国語', 'es': 'スペイン語',
    'fr': 'フランス語', 'de': 'ドイツ語', 'it': 'イタリア語', 'pt': 'ポルトガル語',
    'ru': 'ロシア語', 'ar': 'アラビア語', 'hi': 'ヒンディー語', 'nl': 'オランダ語',
    'tr': 'トルコ語', 'uk': 'ウクライナ語', 'vi': 'ベトナム語', 'th': 'タイ語', 'id': 'インドネシア語',
}


def emit(kind: str, **values) -> None:
    # Escaping non-ASCII keeps the JSON protocol safe even when a caller has
    # inherited a legacy Windows console encoding.
    print(json.dumps({'type': kind, **values}, ensure_ascii=True), flush=True)


def save_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(np.rint(np.clip(samples, -1.0, 1.0) * 32767.0), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def build_speech_regions(intervals: list[tuple[float, float]], start: float,
                         end: float) -> list[tuple[float, float]]:
    """Merge nearby VAD spans, then keep each model input short and bounded."""
    clipped = sorted(
        (max(start, interval_start), min(end, interval_end))
        for interval_start, interval_end in intervals
        if interval_start < end and interval_end > start
    )
    merged: list[list[float]] = []
    for region_start, region_end in clipped:
        if region_end <= region_start:
            continue
        if (merged and region_start - merged[-1][1] <= SPEECH_REGION_MERGE_GAP_SECONDS
                and region_end - merged[-1][0] <= MAX_SPEECH_REGION_SECONDS):
            merged[-1][1] = max(merged[-1][1], region_end)
        else:
            merged.append([region_start, region_end])

    regions: list[tuple[float, float]] = []
    for region_start, region_end in merged:
        duration = region_end - region_start
        pieces = max(1, math.ceil(duration / MAX_SPEECH_REGION_SECONDS))
        for piece in range(pieces):
            piece_start = region_start + duration * piece / pieces
            piece_end = region_start + duration * (piece + 1) / pieces
            regions.append((
                max(start, piece_start - SPEECH_REGION_PADDING_SECONDS),
                min(end, piece_end + SPEECH_REGION_PADDING_SECONDS),
            ))
    return regions


def prepare_local_model(cache_dir: Path) -> Path:
    """Cache model metadata and fetch large weight files in resumable byte ranges."""
    emit('stage', message='モデルファイルを準備しています…')
    snapshot = Path(snapshot_download(
        MODEL_ID,
        cache_dir=str(cache_dir),
        ignore_patterns=['*.safetensors', '*.bin', '*.pt', '*.pth'],
    ))
    index_path = snapshot / 'model.safetensors.index.json'
    if not index_path.is_file():
        raise RuntimeError('MOSSの重み一覧ファイルが見つかりません。')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    weight_names = sorted(set(index.get('weight_map', {}).values()))
    if not weight_names:
        raise RuntimeError('MOSSの重み一覧が空です。')

    for filename in weight_names:
        target = snapshot / filename
        url = hf_hub_url(MODEL_ID, filename)
        head = requests.head(url, allow_redirects=True, timeout=(20, 60))
        head.raise_for_status()
        total_bytes = int(head.headers.get('x-linked-size') or head.headers.get('content-length') or 0)
        if total_bytes <= 0:
            raise RuntimeError(f'MOSS重みのサイズを取得できません: {filename}')
        if target.is_file() and target.stat().st_size == total_bytes:
            continue

        # Each completed part survives cancellation or an interrupted run.
        parts_dir = snapshot.parent / f'.moss-parts-{snapshot.name}' / filename
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_size = 32 * 1024 * 1024
        parts = [(i, start, min(total_bytes, start + part_size) - 1)
                 for i, start in enumerate(range(0, total_bytes, part_size))]
        completed = {}
        for i, start, end in parts:
            part = parts_dir / f'{i:05d}.part'
            expected = end - start + 1
            if part.is_file() and part.stat().st_size == expected:
                completed[i] = expected
        reported_bytes = sum(completed.values())
        progress_lock = threading.Lock()
        emit('download', downloaded=reported_bytes, total=total_bytes, filename=filename)

        def fetch_part(part_info):
            nonlocal reported_bytes
            i, start, end = part_info
            part = parts_dir / f'{i:05d}.part'
            expected = end - start + 1
            if i in completed:
                return
            temporary = part.with_suffix('.part.tmp')
            for attempt in range(3):
                try:
                    with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                                      allow_redirects=True, stream=True, timeout=(20, 120)) as response:
                        if response.status_code != 206:
                            raise RuntimeError(f'重みダウンロードの応答が不正です: HTTP {response.status_code}')
                        content_range = response.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {start}-{end}/'):
                            raise RuntimeError(f'重みダウンロード範囲が一致しません: {content_range}')
                        written = 0
                        with temporary.open('wb') as out:
                            for data in response.iter_content(chunk_size=1024 * 1024):
                                if data:
                                    out.write(data)
                                    written += len(data)
                        if written != expected:
                            raise RuntimeError(f'重みデータが不完全です: {written}/{expected} bytes')
                    os.replace(temporary, part)
                    with progress_lock:
                        reported_bytes += expected
                        emit('download', downloaded=reported_bytes, total=total_bytes, filename=filename)
                    return
                except Exception:
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix='moss-download') as pool:
            futures = [pool.submit(fetch_part, info) for info in parts]
            for future in as_completed(futures):
                future.result()

        partial_target = target.with_suffix(target.suffix + '.tmp')
        emit('stage', message='ダウンロード済みの重みを結合しています…')
        with partial_target.open('wb') as out:
            for i, _, _ in parts:
                with (parts_dir / f'{i:05d}.part').open('rb') as source:
                    shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        if partial_target.stat().st_size != total_bytes:
            raise RuntimeError('結合後のMOSS重みサイズが一致しません。')
        os.replace(partial_target, target)
        shutil.rmtree(parts_dir)

    return snapshot


def main() -> int:
    if len(sys.argv) not in (5, 6, 7):
        print('usage: moss_worker.py INPUT.f32 OUTPUT.json LANGUAGE CHUNK_SECONDS [RESUME_SECONDS [VAD.json]]', file=sys.stderr)
        return 2

    source_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
    language = sys.argv[3]
    chunk_seconds = max(30, int(sys.argv[4]))
    resume_seconds = max(0.0, float(sys.argv[5])) if len(sys.argv) >= 6 else 0.0
    vad_path = Path(sys.argv[6]) if len(sys.argv) == 7 else None
    if language not in LANGUAGES:
        print(f'Unsupported language: {language}', file=sys.stderr)
        return 2

    emit('stage', message='PyTorchとMOSSを初期化しています…')
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
        cache_dir = output_path.parent.parent / 'models' / 'moss-hf'
        model_path = prepare_local_model(cache_dir)
        emit('stage', message=f'{device.type.upper()}上でモデルを読み込んでいます…')
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation='sdpa',
            cache_dir=str(cache_dir),
            local_files_only=True,
        ).to(dtype=dtype).to(device).eval()
        # Greedy decoding can get stuck repeating a partial word (for example
        # "d- d- d-"), preventing EOS and consuming the entire token budget.
        # A mild penalty fixes this on the reported clip without changing the
        # transcript format or forcing shorter audio chunks.
        model.generation_config.repetition_penalty = MOSS_REPETITION_PENALTY
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )

        samples_in_file = source_path.stat().st_size // np.dtype('<f4').itemsize
        start_sample = int(round(resume_seconds * SAMPLE_RATE))
        if start_sample > samples_in_file:
            raise RuntimeError('再開位置が音声の長さを超えています。')
        total_seconds = samples_in_file / SAMPLE_RATE
        block_samples = chunk_seconds * SAMPLE_RATE
        total_chunks = max(1, math.ceil(samples_in_file / block_samples))
        final_tail_samples = samples_in_file % block_samples
        merge_short_final_tail = (
            total_chunks > 1
            and 0 < final_tail_samples <= MIN_RETRY_SECONDS * SAMPLE_RATE
        )
        if merge_short_final_tail:
            # A tiny final block (for example 3.1 s after a five-hour file)
            # is too little context for MOSS to reliably produce timestamped
            # output. Process it with the preceding full block instead.
            total_chunks -= 1
        first_chunk = start_sample // block_samples
        if first_chunk * block_samples != start_sample:
            raise RuntimeError('MOSSの再開位置が処理区間の境界ではありません。')
        prompt = (
            '请将音频转写为文本，每一段需以起始时间戳和说话人编号'
            '（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，'
            '并在段末标注结束时间戳，以清晰标明该段语音范围。'
        )
        if language != 'auto':
            prompt += f'音频语言是{LANGUAGES[language]}，请使用该语言转写。'
        all_segments = []
        speech_intervals = None
        if vad_path is not None:
            vad_data = json.loads(vad_path.read_text(encoding='utf-8'))
            speech_intervals = [
                (max(0.0, float(item['start'])), max(0.0, float(item['end'])))
                for item in vad_data.get('speech_intervals', [])
                if float(item.get('end', 0.0)) > float(item.get('start', 0.0))
            ]

        def transcribe_audio(samples: np.ndarray, offset: float, chunk_index: int,
                             total_chunks: int, depth: int = 0) -> list[dict]:
            """Transcribe a chunk, splitting it if output is empty or truncated."""
            if not samples.size:
                return []

            duration = samples.size / SAMPLE_RATE
            if speech_intervals is not None and not any(
                    start < offset + duration + 0.5 and end > offset - 0.5
                    for start, end in speech_intervals):
                return []

            # Genuine digital silence has no transcript to recover. Avoid
            # repeatedly invoking the model on silent tails after subdivision.
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
            if peak < SILENCE_PEAK_THRESHOLD and rms < SILENCE_RMS_THRESHOLD:
                return []

            chunk_path = output_path.with_name(
                f'{output_path.stem}.{chunk_index + 1}.retry-{depth}.wav')
            try:
                save_wav(chunk_path, samples)
                messages = build_transcription_messages(chunk_path, prompt=prompt)
                last_reported = 0

                def token_progress(count: int) -> None:
                    nonlocal last_reported
                    if count - last_reported >= 25:
                        emit('tokens', index=chunk_index + 1, total=total_chunks,
                             count=count, retry=depth)
                        last_reported = count

                result = generate_transcription(
                    model,
                    processor,
                    messages,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    device=device,
                    dtype=dtype,
                    token_callback=token_progress,
                )
                generated_tokens = int(result.get('generated_tokens', 0))
                segments = parse_transcript(result['text'])
                parsed = []
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        start = float(segment.start) + offset
                        parsed.append({
                            'start': max(0.0, start),
                            'end': max(start + 0.2, float(segment.end) + offset),
                            'text': text,
                            'speaker': segment.speaker,
                        })

                hit_token_limit = generated_tokens >= MAX_NEW_TOKENS - 8
                # Only token-limit exhaustion triggers subdivision. A chunk
                # that simply cannot be parsed is skipped so it cannot cause
                # repeated inference or fragment the surrounding audio.
                needs_retry = hit_token_limit
                can_split = duration > MIN_RETRY_SECONDS + 0.01
                if needs_retry and can_split:
                    midpoint = samples.size // 2
                    if midpoint > 0 and midpoint < samples.size:
                        emit('stage', message=(
                            f'区間 {chunk_index + 1}/{total_chunks} は生成上限に達したため、'
                            f'{duration:.0f}秒から分割して再試行します…'))
                        return (
                            transcribe_audio(samples[:midpoint], offset, chunk_index,
                                             total_chunks, depth + 1)
                            + transcribe_audio(samples[midpoint:],
                                               offset + midpoint / SAMPLE_RATE,
                                               chunk_index, total_chunks, depth + 1)
                        )

                if hit_token_limit:
                    if parsed:
                        emit('stage', message=(
                            f'{offset:.1f}秒付近の短い区間でも生成上限に達しました。'
                            '生成済みの字幕を保存して次の区間へ進みます…'))
                        return parsed
                    emit('warning', message=(
                        f'{offset:.1f}秒付近の短い区間は字幕を解析できなかったため、'
                        'この区間をスキップして続行します…'))
                    return []
                if not parsed:
                    message = (
                        f'{offset:.1f}〜{offset + duration:.1f}秒の区間を字幕として解析できませんでした。'
                        'この区間を字幕なしとして記録し、残りの区間を続行します。')
                    emit('warning', message=message)
                    return []
                return parsed
            finally:
                chunk_path.unlink(missing_ok=True)

        with source_path.open('rb') as source:
            source.seek(start_sample * np.dtype('<f4').itemsize)
            for chunk_index in range(first_chunk, total_chunks):
                read_samples = block_samples
                if merge_short_final_tail and chunk_index == total_chunks - 1:
                    read_samples += final_tail_samples
                samples = np.fromfile(source, dtype='<f4', count=read_samples)
                if not samples.size:
                    break
                offset = chunk_index * chunk_seconds
                chunk_end = offset + samples.size / SAMPLE_RATE
                has_speech = speech_intervals is None or any(
                    start < chunk_end + 0.5 and end > offset - 0.5
                    for start, end in speech_intervals
                )
                if has_speech:
                    if speech_intervals is None:
                        chunk_segments = transcribe_audio(
                            samples, offset, chunk_index, total_chunks)
                    else:
                        chunk_segments = []
                        regions = build_speech_regions(speech_intervals, offset, chunk_end)
                        for region_start, region_end in regions:
                            local_start = max(0, int(round((region_start - offset) * SAMPLE_RATE)))
                            local_end = min(samples.size, int(round((region_end - offset) * SAMPLE_RATE)))
                            if local_end <= local_start:
                                continue
                            emit('stage', message=(
                                f'{region_start:.0f}〜{region_end:.0f}秒の発話を文字起こし中…'))
                            chunk_segments.extend(transcribe_audio(
                                samples[local_start:local_end], region_start,
                                chunk_index, total_chunks))
                else:
                    chunk_segments = []
                    emit('stage', message=(
                        f'{offset:.0f}〜{chunk_end:.0f}秒は発話が検出されず、MOSSをスキップします…'))
                all_segments.extend(chunk_segments)

                processed = (total_seconds if merge_short_final_tail and chunk_index == total_chunks - 1
                             else min(total_seconds, (chunk_index + 1) * chunk_seconds))
                emit('chunk', index=chunk_index + 1, total=total_chunks,
                     processed=processed,
                     segments=chunk_segments)

        output_path.write_text(json.dumps({'segments': all_segments}, ensure_ascii=False), encoding='utf-8')
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
