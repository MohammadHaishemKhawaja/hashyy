@echo off
REM ============================================================
REM  Pick a profile by setting PROFILE to FAST or QUALITY
REM ============================================================
set PROFILE=FAST

set CUDA_VISIBLE_DEVICES=0
mkdir "%~dp0kvcache" 2>nul

if /I "%PROFILE%"=="FAST" (
  REM IQ3_XXS 3-bit  ~63 tok/s  10.8GB VRAM   speculation OFF on purpose
  REM   ncmoe 7 = 72 tok/s but 11.5GB - too tight if you use the desktop.
  set MODEL=Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
  set NCMOE=10
  set SPEC=
) else (
  REM Q4_K_M 4-bit  ~45 tok/s  10.9GB VRAM   speculation ON (pays off here)
  set MODEL=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  set NCMOE=26
  set SPEC=--spec-type draft-mtp --spec-draft-n-max 3
)

"%~dp0llama\llama-server.exe" ^
  -m "%~dp0models\%MODEL%" ^
  -ngl 99 -ncmoe %NCMOE% -fa on -ctk q8_0 -ctv q8_0 -ub 256 -c 8192 ^
  %SPEC% --slot-save-path "%~dp0kvcache" --jinja ^
  --host 127.0.0.1 --port 8080
