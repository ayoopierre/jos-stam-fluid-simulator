#include <cstdio>
#include <thread>

#include "fluid_cpu.hpp"
#include "fluid_gpu.hpp"
#include "window_wrapper.hpp"

int main(void){
    FluidCpu f(400);
    FluidGpu f2(400);
    Window w(800, 800, &f2);

    while (!w.shouldClose())
    {
        f2.step(0.00005);
        w.update();
    }   
}
