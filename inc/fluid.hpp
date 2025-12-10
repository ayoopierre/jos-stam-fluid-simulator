#ifndef MY_FLUID
#define MY_FLUID

#include "utils.hpp"

class Fluid
{
public:
    Fluid() = default;
    virtual void step(float dt) = 0;
    virtual bool draw_into_bitmap(MyBitmap& bitmap) = 0;
};

#endif