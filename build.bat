cl /EHsc /MD /std:c++17 /O2 /I"inc" /I"C:\Users\Mikim\Documents\Tools\raylib-5.5_win64_msvc16\include" src\*.cpp /Fo:build\ /Fe:main.exe ^
    /link "fluid_solver_gpu.lib" "C:\Users\Mikim\Documents\Tools\raylib-5.5_win64_msvc16\lib\raylib.lib" ^
    User32.lib Gdi32.lib Winmm.lib Kernel32.lib Ole32.lib Shell32.lib
