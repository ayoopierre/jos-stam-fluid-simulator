#include <cstdio>
#include <thread>

#include "fluid_cpu.hpp"
#include "fluid_gpu.hpp"
#include "window_wrapper.hpp"

int main(void){
    FluidCpu f(200);
    FluidGpu f2(200);
    Window w(800, 800, &f);

    f2.step(0.0f);

    while (!w.shouldClose())
    {
        f.step(0.0005);
        w.update();
        w.handle_input();
    }   
}
