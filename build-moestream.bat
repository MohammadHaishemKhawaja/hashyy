@echo off
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%
set CUDA_PATH=%ROOT%\build-tools\cuda\_root
set CUDAToolkit_ROOT=%CUDA_PATH%
set PATH=%ROOT%\build-tools\cmake-4.4.3-windows-x86_64\bin;%ROOT%\build-tools;%CUDA_PATH%\bin;%PATH%
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%ROOT%\src\lcpp"
echo === checking out PR branch ===
git checkout tryms
echo === configuring ===
cmake -B build-ms -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120 ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DGGML_NATIVE=ON
if errorlevel 1 ( echo CONFIGURE FAILED & exit /b 1 )
echo === building ===
cmake --build build-ms --target llama-server llama-bench -j 6
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
echo === DONE ===
dir /b build-ms\bin\*.exe
