nvcc -std=c++17 -Xcompiler "/std:c++17" ^
    -I"inc" ^
    -lib -o fluid_solver_gpu.lib cusrc/fluid_gpu.cu ^
    --compiler-options "/EHsc /MD" ^
    -lcudadevrt -lcudart


cl /EHsc /MD /std:c++17 /O2 /I"inc" /I"C:\Users\Mikim\Documents\Tools\raylib-5.5_win64_msvc16\include"^
    /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" src\*.cpp /Fo:build\ /Fe:main.exe ^
    /link "fluid_solver_gpu.lib" "C:\Users\Mikim\Documents\Tools\raylib-5.5_win64_msvc16\lib\raylib.lib" ^
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cuda.lib" ^
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudart.lib" ^
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\cudadevrt.lib" ^
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\nvfatbin.lib" ^
    User32.lib Gdi32.lib Winmm.lib Kernel32.lib Ole32.lib Shell32.lib
