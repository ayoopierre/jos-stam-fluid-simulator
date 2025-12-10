#ifndef WINDOW_H
#define WINDOW_H

#include <cmath>
#include <raylib.h>

#include "fluid_cpu.hpp"
#include "utils.hpp"

class Window
{
private:
    size_t width, height;
    Texture2D tex;
    Image img;
    Color *pixels;
    MyBitmap bitmap;
    Fluid *f;

public:
    Window(int width, int height, Fluid *f)
    {
        InitWindow(width, height, "My window");
        this->width = width;
        this->height = height;
        img = GenImageColor(width, height, BLANK);
        tex = LoadTextureFromImage(img);
        pixels = LoadImageColors(img);

        bitmap = { (MyColor *)pixels, this->width, this->height, tex.id };
        this->f = f;
        SetTargetFPS(60);
    }

    inline bool shouldClose()
    {
        return WindowShouldClose();
    }

    inline void update()
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        /* Trying to use fluid update method */
        if(f->draw_into_bitmap(bitmap)){
            UpdateTexture(tex, pixels);
        }

        DrawTexturePro(tex,
                       {0, 0, (float)width, (float)height},
                       {0, 0, (float)this->width, (float)this->height},
                       {0, 0},
                       0,
                       WHITE);

        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, GREEN);
        EndDrawing();
    }

    inline void handle_input()
    {
        static Vector2 prev = {0};
        Vector2 mouse_pos = GetMousePosition();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        {
            // int x = (int)(mouse_pos.x * f->N / this->width);
            // int y = (int)(mouse_pos.y * f->N / this->height);

            // float dx = (mouse_pos.x - prev.x) / (float)this->width;
            // float dy = (mouse_pos.y - prev.y) / (float)this->height;

            // f->add_fluid(x, y, 0.1);
            // f->add_u(x, y, dx);
            // f->add_v(x, y, -dy);
        }
        prev = mouse_pos;
    }

    inline Color HeatColor(float t)
    {
        return {
            (unsigned char)(255 * t), // red
            (unsigned char)(255 * t), // green peak in middle
            (unsigned char)(255 * t), // blue
            255};
    }

    inline void close()
    {
        CloseWindow();
    }
};

#endif