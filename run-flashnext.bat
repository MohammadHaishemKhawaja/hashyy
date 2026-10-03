@echo off
REM Qwen3.8-Flash-Next REAP128 (86B, 128 of 512 experts) - 14.4 tok/s on 5070 + 32GB RAM
REM Launched through safe_run.py: refuses RAM-exhausting flags, kills server if RAM < 0.8GB.
REM
REM STRONG CAVEAT: this build is REAP-pruned on code/agent data. Code, math and
REM reasoning are good. WORLD KNOWLEDGE IS DAMAGED (it cannot name Canberra).
REM Use it for coding. Use Qwen3.6 (run.bat) for general questions.
set CUDA_VISIBLE_DEVICES=0
mkdir "%~dp0kvcache" 2>nul
python "%~dp0scripts\safe_run.py" ^
  -m "%~dp0models\reap128\Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-UD-IQ4_XS.gguf" ^
  --floor-gb 0.8 --strikes 8 ^
  --load-mode mmap --lazy-mode on ^
  -ngl 99 -ncmoe 36 -ub 256 -t 6 ^
  -fa on -ctk q8_0 -ctv q8_0 -c 4096 ^
  --slot-save-path "%~dp0kvcache" --jinja ^
  --host 127.0.0.1 --port 8080
