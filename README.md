Parakeet SRT Transcriber

Updates
- Large files are uploaded as a raw stream and written to disk in small pieces. This avoids the extra full-size temporary copy created by multipart uploads.
- Upload progress is shown in the browser, and an upload can be canceled.
- MOSS uses the selected spoken language (English is the default; automatic detection is also available). Parakeet v3 detects the language automatically and does not use the language selector; it supports 25 European languages, not Japanese, Chinese, or Korean.
- A file path on the server PC can be entered or chosen with the native file picker to transcribe directly, without uploading it. The original file is kept. The picker requires the server to run in the logged-in desktop session.
- On Windows, use “Colab転送用に音声だけを抽出” to save only the selected file's first audio track in an MKA container without re-encoding the audio stream. Upload that much smaller MKA to Colab's `/content` file manager, then transcribe its `/content/...mka` path. FFmpeg reports extraction progress; the original media is unchanged. This is useful when uploading a multi-gigabyte video directly to Colab would take too long.
- The recognized text is shown progressively after each audio chunk has been processed, then replaced by the final SRT text when complete.
- Choose an optional transcription range with HH:MM:SS or seconds. The generated subtitle timestamps stay aligned to the original media timeline.
- Incomplete jobs save extracted PCM and completed transcript chunks under `tmp\checkpoints`. Resume saved work from the browser after an error, cancellation, or server restart. Parakeet checkpoints every 4 seconds; MOSS checkpoints after each 1-minute section, so a partially processed MOSS section is repeated on resume.
- MOSS uses a mild repetition penalty during greedy decoding to prevent occasional partial-word loops (such as repeating "d-" indefinitely) from consuming the full token budget. The configured output limit is 5120 tokens on both Windows and Colab; the existing recovery behavior remains available if generation still reaches its limit.
- A final MOSS audio tail of 4 seconds or less is joined to the preceding block. If repeated subdivision still isolates an unparseable, non-speech tail, the job completes with a warning instead of failing the entire transcription.
- MOSS uses Silero VAD to avoid sending speech-free sections to the model. With VAD enabled, speech spans are grouped into regions up to 16 seconds, with a 2.5-second merge gap. With VAD disabled, the selected 1-60 minute audio blocks are kept intact whenever speech is present; only blocks with no detected speech (such as music-only sections) are skipped. The VAD runs on one CPU thread. If output reaches the token limit (5120 on both Windows and Colab), MOSS subdivides down to 4-second spans and continues after salvaging any parseable text in a pathological short span.
- When VAD is disabled, set the MOSS processing interval in the UI to 1-60 minutes. The default and recommended starting point is 1 minute for shorter recovery intervals; try 2-3 minutes when more surrounding context is useful. This setting is saved with the checkpoint and retained when resuming.
- Resume data stays on disk until the job completes or you delete it from the checkpoint list. Noise-suppressed audio is retained instead of storing both the original PCM and enhanced PCM.
- Each subtitle cue stays on screen for at least 3 seconds. Cues may overlap when needed to meet this minimum.
- The downloaded SRT keeps the source media's base filename and uses the .srt extension.
- Optional speech enhancement can run before transcription to suppress background noise and sound effects. GTCRN uses Sherpa-ONNX's offline API in 30-second blocks, with one worker per physical CPU core (hyper-threaded/virtual cores are excluded). DPDFNet baseline remains available on CUDA when stronger suppression is preferred. Models download on first use. The install script adds the CUDA-enabled Sherpa-ONNX runtime (about 153 MB on Windows).
- FFmpeg audio extraction reports converted audio time while it runs. The CUDA startup diagnostic no longer enables ONNX Runtime profiling, so it does not leave profile JSON files in the application folder.
- MOSS-Transcribe-Diarize is an optional ASR model choice. Run `install_moss_windows.bat` once to install its isolated Python/CUDA runtime; model weights download on first use. It uses CUDA 12.6 PyTorch wheels to retain support for older Pascal GPUs such as GTX 1080. With VAD enabled, it processes short speech regions; with VAD disabled, it processes the selected 1-60 minute blocks that contain speech. It includes speaker-aware timestamps and can be slower than Parakeet on older GPUs.
- MOSS language choices include automatic detection (no language hint) plus Japanese, English, Chinese, Korean, Spanish, French, German, Italian, Portuguese, Russian, Arabic, Hindi, Dutch, Turkish, Ukrainian, Vietnamese, Thai, and Indonesian. The selector labels are shown in Japanese, and English is selected by default.

Run
1. Stop any older server window for this application.
2. Run install_windows.bat to install/update dependencies, then run start_windows.bat.
3. Open http://127.0.0.1:5173. Select a video to upload it, or choose/enter a path on the server PC to process it directly. Parakeet detects the spoken language automatically; when using MOSS, choose the spoken language before starting.

Large uploads require enough free disk space in this application's tmp folder for the original media and the extracted audio. Upload time depends on available disk speed.

Google Colab
1. Open [the Colab notebook](https://colab.research.google.com/github/Elaina117/Parakeet_SRT_Transcriber/blob/main/Parakeet_SRT_Transcriber_Colab.ipynb) and select a GPU runtime.
2. Upload the media file (or the Windows-extracted MKA audio) through Colab's file manager into /content. Do not upload an application ZIP; the notebook clones the latest app code from this public GitHub repository and pulls updates every time the cell starts.
3. Run the single code cell. On first run it installs the Linux/CUDA 12 dependencies, MOSS, and speech enhancement, then starts the app and displays its page. Re-running in the same live runtime overlays updated app files without deleting checkpoints/models, and skips installation when the setup fingerprint is unchanged. The cell remains active and checks the server every few minutes while the app is running; stop the cell to shut down the app. This does not override Colab runtime limits or forced disconnections.
4. Enter a local media path such as `/content/movie.mp4` or `/content/movie.audio.mka` in the direct-processing field. Nothing uses or mounts Google Drive. App files, models, temporary audio, and checkpoints are deleted when the Colab runtime itself is discarded.

GitHub source updates
- Edit the shared app files in this repository; Windows uses the working files and Colab fetches them from `main` when its cell starts.
- In GitHub Desktop, review changed files, enter a short commit summary, commit to `main`, then click `Push origin`. The public Colab notebook will use that pushed version the next time the cell runs.
- Runtime state, models, media, logs, generated ZIP bundles, and the nested extracted app copy are excluded from the GitHub repository by `.gitignore`.
