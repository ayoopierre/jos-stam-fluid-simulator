nvcc -std=c++17 -Xcompiler "/std:c++17" ^
    -I"inc" -I"cuinc" ^
    -I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um" ^
    -I"C:\Users\Mikim\Documents\Tools\raylib-5.5_win64_msvc16\include" ^
    -lib -o fluid_solver_gpu.lib cusrc/fluid_gpu.cu cusrc/minmax.cu ^
    --compiler-options "/EHsc /MD" ^
    -lcudadevrt -lcudart -lopengl32