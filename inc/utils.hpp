#ifndef MY_UTILS
#define MY_UTILS

#include <cstdlib>
#include <raylib.h>

#ifndef MY_DBL_MAX
#define MY_DBL_MAX double(1.79769313486231570814527423731704357e+308L)
#endif

#ifndef MY_DBL_MIN
#define MY_DBL_MIN double(2.22507385850720138309023271733240406e-308L)
#endif

struct MyBitmap
{
    Color *bitmap;
    size_t width;
    size_t height;
};

typedef MyBitmap MyBitmap;

struct MyTile
{
    int start_x;
    int start_y;
    int width;
    int height;
};

typedef MyTile MyTile;

#endif