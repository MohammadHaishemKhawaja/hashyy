@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
echo [vcvars exit] %errorlevel%
echo [WindowsSdkDir] %WindowsSdkDir%
echo [WindowsSDKVersion] %WindowsSDKVersion%
where cl 2>nul || echo cl.exe NOT FOUND
set CUDA_PATH=%~dp0build-tools\cuda\_root
set PATH=%CUDA_PATH%\bin;%PATH%
where nvcc 2>nul || echo nvcc NOT FOUND
echo __global__ void k(){} > "%TEMP%\t.cu"
echo [nvcc test]
nvcc -arch=sm_120 -c "%TEMP%\t.cu" -o "%TEMP%\t.obj" 2>&1
echo [nvcc exit] %errorlevel%
