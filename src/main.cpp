#include <cstdio>
#include <thread>

#include "fluid.hpp"
#include "window_wrapper.hpp"

int main(void){
    Fluid f(100);
    Window w(800, 800, &f);

    // f.step(0.001);

    while (!w.shouldClose())
    {
        f.step(0.001);
        w.update();
        w.handle_input();
    }   
}
