#ifndef FLUID_HPP
#define FLUID_HPP

#define GS_ITER 30

#include <vector>
#include <cmath>
#include <cstdio>

class Window;

class Fluid
{
public:
    Fluid(size_t N);

    inline void step(double dt)
    {
        density_step(dt);
    }

    inline void density_step(double dt)
    {
        diffuse_density(dt);
        advect_density(dt);
    }

    inline void velocity_step(double dt)
    {
    }

    inline void add_fluid(int x, int y, double val)
    {
        density[at(x, y)] += val;
    }

    constexpr size_t at(size_t x, size_t y) noexcept { return y * (N + 2) + x; };

private:
    friend Window;

    enum class BoundaryHandleEnum
    {
        HandleU,
        HandleV,
        HandleRho
    };

    void apply_forces(double dt);
    void diffuse_density(double dt);
    void diffues_velocity(double dt);
    void project();
    void advect_field(double dt, double *input_field, double *output_field);
    void advect_density(double dt);
    void advect_velocity(double dt);
    void handle_boundaries(enum BoundaryHandleEnum e, double *data);

    /* Gauss-Seidel solver for Laplace equation arising from diffusion terms */
    void laplace_eq_GS_solver(double *x, double *b, double a, double c, enum BoundaryHandleEnum e);
    /* Sample discrete field using continous coos, using bilinear interpolation */
    double sample_field(double x, double y, double *field);

    std::vector<double> density;
    std::vector<double> u;
    std::vector<double> v;

    std::vector<double> new_u;
    std::vector<double> new_v;
    std::vector<double> new_density;

    std::vector<bool> is_wall;
    std::vector<double> divergence;
    std::vector<double> p;

    size_t N;

    double box_length = 2.0;
    double h;

    double diff = 0.01;
    double visc;
};

#endif