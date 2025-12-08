#ifndef MY_UTILS
#define MY_UTILS

#include <cstdlib>

#ifndef MY_DBL_MAX
#define MY_DBL_MAX float(1.79769313486231570814527423731704357e+308L)
#endif

#ifndef MY_DBL_MIN
#define MY_DBL_MIN float(2.22507385850720138309023271733240406e-308L)
#endif

struct MyColor{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

typedef MyColor MyColor;

struct MyBitmap
{
    MyColor *bitmap;
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