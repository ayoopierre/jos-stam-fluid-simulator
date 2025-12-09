#ifndef FLUID_CPU_HPP
#define FLUID_CPU_HPP

#define GS_ITER 30

#include <vector>
#include <cmath>
#include <cstdio>

#include "fluid.hpp"
#include "utils.hpp"

class Window;

class FluidCpu : public Fluid
{
public:
    FluidCpu(size_t N);

    inline void step(float dt)
    {
        apply_sources(dt);
        density_step(dt);
        velocity_step(dt);
    }

    inline void density_step(float dt)
    {
        diffuse_density(dt);
        advect_density(dt);
    }

    inline void velocity_step(float dt)
    {
        diffues_velocity(dt);
        project();
        advect_velocity(dt);
        project();
    }

    inline void add_fluid(int x, int y, float val)
    {
        density[at(x, y)] += val;
    }

    inline void add_u(int x, int y, float val)
    {
        density[at(x, y)] += val;
    }

    inline void add_v(int x, int y, float val)
    {
        density[at(x, y)] += val;
    }

    void draw_into_bitmap(MyBitmap &bitmap);

    constexpr size_t at(size_t x, size_t y) noexcept { return y * (N + 2) + x; };

    constexpr MyColor heat_color(float t) noexcept
    {
        return {
            (unsigned char)(255 * t), // red
            (unsigned char)(255 * t), // green peak in middle
            (unsigned char)(255 * t), // blue
            255};
    }

private:
    friend Window;

    enum class BoundaryHandleEnum
    {
        HandleU,
        HandleV,
        HandleRho
    };

    void apply_sources(float dt);
    // void apply_forces(float dt);
    void diffuse_density(float dt);
    void diffues_velocity(float dt);
    void project();
    void advect_field(float dt, float *input_field, float *output_field);
    void advect_density(float dt);
    void advect_velocity(float dt);
    void handle_boundaries(enum BoundaryHandleEnum e, float *data);

    /* Gauss-Seidel solver for Laplace equation arising from diffusion terms */
    void laplace_eq_GS_solver(float *x, float *b, float a, float c, enum BoundaryHandleEnum e);
    /* Sample discrete field using continous coos, using bilinear interpolation */
    float sample_field(float x, float y, float *field);

    std::vector<float> density;
    std::vector<float> u;
    std::vector<float> v;

    std::vector<float> new_u;
    std::vector<float> new_v;
    std::vector<float> new_density;

    std::vector<bool> is_wall;
    std::vector<float> divergence;
    std::vector<float> p;

    size_t N;

    float box_length = 1.0;
    float h;

    float diff = 1.01;
    float visc = 0.01;
};

#endif