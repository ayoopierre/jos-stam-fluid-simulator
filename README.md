# About
Simple real-time fluid simulations based on Jos Stam paper "Stable fluids".
Implementation contains CPU version as well as GPU accelerated one.

![](example.gif)

# Building
This application has following dependencies:
- CUDA 
- Raylib

**Note on Raylib** - this library was used simply for the fact how easy it 
is to setup simple window and access underlying texture of the window (both
for CPU and GPU). Since we want to avoid unnecesery CPU and GPU transactions
over PCIe fabric during simulation on GPU, we want to access underlying window
texture that is stored directly on the GPU. Luckily since Raylib is a simple 
wrapper around OpenGL this can be done as CUDA already is shipped with OpenGL
interop functionalities. Now GPU solver, will directlly render into OpenGL
texture created by Raylib context, in GPU code only OpenGL API is used. This is
why need to link against OpenGL while compiling GPU code into a static library.

**Note on building static library from CUDA** - NVCC compiler will compile
all functions marked with `__host__` attribute using host native compiler
(MSVC compiler on Windows), and use CUDA compiler for functions marked with
`__device__` attribute. Functions marked `__host__` attribute will land in 
normal code sections, meanwhile `__device__` code after being compiled to 
PTX format will be placed in .nvfatbin section (during runtime data in this
section will be compiled by runtime/driver into compatible device code). 
Host code requires some OpenGL definitions but those are shipped with MSVC
compiler already and include path and path to libraries should be resolved
automatically by NVCC compiler during installation (NVCC will use MSVC resources).

[https://hpcgpu.mini.pw.edu.pl/cuda-compilation-toolchain/]


## Instaling dependencies
1. MSVC is automatically installed with Visual Studio. This instalation
should already contain OpenGL header files and static libraries to link
against.
2. Using CUDA toolkit instalator for Windows will already install
appropriate version for MSVC compiler since host side code for 
CUDA projects is already compiled by host native compiler (which
is MSVC in this case)
3. Raylib 5.5 can be installed from official Github realese:
[https://github.com/raysan5/raylib/releases/tag/5.5]
Select version appropriate for the compiler, for Windows with
MSVC compiler this would be:
![](image.png)

## Building scripts
Commands needed to compile the project are present in `build_cpu.bat`
and `build_gpu.bat`. There is also one joined script that will call both
of those. To use this script fill out path to unzipped Raylib directory
in build.bat file. Launch x64 Native Tools CMD (it is a CMD shipped with
Visual Studio that sets some enviroment variables for building - this step
is important and if skipped build script will not work). Change working
directory to the project directory and launch build.bat script.

## Minimal step to build project
Those steps assume Visual Studio and CUDA already installed. 
1. Download and unzip Raylib realese for MSVC compiler (x64 version)
2. Modify path to Raylib resources in the build.bat script
3. Launch x64 native tools console (shipped with MSVC compiler)
![](image-2.png)
4. Change working directory to the project directory
5. Run build script