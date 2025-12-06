#include <cstdio>
#include <thread>

#include "fluid.hpp"
#include "window_wrapper.hpp"

int main(void){
    Fluid f(400);
    Window w(800, 800, &f);

    // f.step(0.001);

    while (!w.shouldClose())
    {
        f.step(0.0001);
        w.update();
        // std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }   
}
