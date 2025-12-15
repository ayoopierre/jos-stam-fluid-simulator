cl /EHsc /MD /std:c++17 /O2 /I"inc" /I"%RAYLIB_PATH%\include"^
    /I"%CUDA_PATH%\include" src\*.cpp /Fe:main.exe ^
    /link "fluid_solver_gpu.lib" "%RAYLIB_PATH%\lib\raylib.lib" ^
    "%CUDA_PATH%\lib\x64\cuda.lib" ^
    "%CUDA_PATH%\lib\x64\cudart.lib" ^
    "%CUDA_PATH%\lib\x64\cudadevrt.lib" ^
    "%CUDA_PATH%\lib\x64\nvfatbin.lib" ^
    User32.lib Gdi32.lib Winmm.lib Kernel32.lib Ole32.lib Shell32.lib
