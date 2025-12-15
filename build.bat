@echo off

set RAYLIB_PATH=""

call build_gpu_lib.bat
call build_cpu.bat

call rm main.obj
call rm fluid_cpu.obj
