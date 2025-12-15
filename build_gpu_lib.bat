nvcc -std=c++17 -Xcompiler "/std:c++17" ^
    -I"inc" -I"cuinc" ^
    -I "%RAYLIB_PATH%\include" ^
    -lib -o fluid_solver_gpu.lib cusrc/fluid_gpu.cu cusrc/minmax.cu ^
    --compiler-options "/EHsc /MD" ^
    -lcudadevrt -lcudart -lopengl32