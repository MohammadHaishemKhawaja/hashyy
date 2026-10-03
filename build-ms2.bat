@echo off
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%
set CUDA_PATH=%ROOT%\build-tools\cuda\_root
set CUDAToolkit_ROOT=%CUDA_PATH%
set PATH=%ROOT%\build-tools\cmake-4.4.3-windows-x86_64\bin;%ROOT%\build-tools;%CUDA_PATH%\bin;%PATH%
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%ROOT%\src\lcpp"
cmake --build build-ms --target llama-server -j 6
