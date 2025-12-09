# Building
This application has following dependencies:
- CUDA
- Raylib

There is build script which can be refrenced. Build script works
on Windows with installed Raylib, CUDA and MSVC compiler. To compile
following steps have to be made:
1. Fill out path to CUDA libraries and include directories (find path
to nvcc compiler - use which or command similar to that)
2. Fill out path to Raylib include path and path to libraries
3. Create static libary from /cusrc files (use nvcc and similar
command to one used in build.bat script)
4. Build host side code, and link against all libraries that are dependencies
for Raylib, CUDA static libraries listed in build script.

To use build script on Windows machine with installed dependencies,
launch MSVC compiler console (x64 native tools CMD), change working directory
to the one with project code, and launch build.bat script.