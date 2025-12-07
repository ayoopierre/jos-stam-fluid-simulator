#ifndef WINDOW_H
#define WINDOW_H

#include <cmath>

#include "fluid.hpp"
#include "raylib.h"

class Window
{
private:
    int width, height;
    Texture2D tex;
    Image img;
    Color *pixels;
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

        double minVal = __DBL_MAX__, maxVal = __DBL_MIN__;

        for (int y = 0; y < f->N + 2; y++)
        {
            for (int x = 0; x < f->N + 2; x++)
            {
                double v = f->density[f->at(x, y)];
                if (v > maxVal)
                    maxVal = v;
                if (v < minVal)
                    minVal = v;
            }
        }

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double f_x = f->box_length * x / (double)width;
                double f_y = f->box_length * y / (double)height;
                // printf("%lf, %lf\n", f_x, f_y);
                double v = f->sample_field(f_x, f_y, f->density.data());
                double t = (v - minVal) / (maxVal - minVal);
                pixels[y * width + x] = HeatColor(t);
            }
        }

        UpdateTexture(tex, pixels);

        DrawTexturePro(tex,
                       (Rectangle){0, 0, (float)width, (float)height},
                       (Rectangle){0, 0, (float)this->width, (float)this->height},
                       (Vector2){0, 0},
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
            int x = (int)(mouse_pos.x * f->N / this->width);
            int y = (int)(mouse_pos.y * f->N / this->height);

            float dx = (mouse_pos.x - prev.x) / (float)this->width;
            float dy = (mouse_pos.y - prev.y) / (float)this->height;

            f->add_fluid(x, y, 0.1);
            f->add_u(x, y, dx);
            f->add_v(x, y, -dy);
        }
        prev = mouse_pos;
    }

    inline Color HeatColor(float t)
    {
        // Blue -> Cyan -> Green -> Yellow -> Red
        // return (Color){
        //     (unsigned char)(255 * t),                // red
        //     (unsigned char)(255 * (1.0f - fabsf(t-0.5f)*2)), // green peak in middle
        //     (unsigned char)(255 * (1.0f - t)),        // blue
        //     255
        return (Color){
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