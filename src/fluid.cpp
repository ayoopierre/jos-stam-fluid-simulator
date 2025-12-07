#include <algorithm>

#include "fluid.hpp"

Fluid::Fluid(size_t N)
{
    this->N = N;
    h = box_length / N;

    density.resize((N + 2) * (N + 2));
    u.resize((N + 2) * (N + 2));
    v.resize((N + 2) * (N + 2));

    new_density.resize((N + 2) * (N + 2));
    new_u.resize((N + 2) * (N + 2));
    new_v.resize((N + 2) * (N + 2));

    is_wall.resize((N + 2) * (N + 2));
    divergence.resize((N + 2) * (N + 2));
    p.resize((N + 2) * (N + 2));

    auto l = [](double &v)
    { v = (rand() / (double)RAND_MAX) - 0.5; };

    auto l1 = [](double &v)
    { v = 0.0; };
    std::for_each(u.begin(), u.end(), l1);
    std::for_each(v.begin(), v.end(), l1);

    std::fill(density.begin(), density.end(), 0.0);

    std::fill(new_u.begin(), new_u.end(), 0.1);
    std::fill(new_v.begin(), new_v.end(), 0.1);
    std::fill(new_density.begin(), new_density.end(), 1.0);
}

void Fluid::apply_sources(double dt)
{
    // density[at(N / 2, N / 2)] += 0.3 * dt;
    for(int y = 4 * N / 10; y <= 6 * N / 10; y++){
        v[at(1, y)] = 0.0;
        u[at(1, y)] = 7.0;
        density[at(1, y)] += 1.0;
    }
}

// void Fluid::apply_forces(double dt)
// {
//     for (int i = 1; i <= N; i++)
//     {
//         u[at(1, i)] += 1.0 * dt;
//     }
// }

void Fluid::diffuse_density(double dt)
{
    double a = diff * dt * (1.0 / h) * (1.0 / h);

    laplace_eq_GS_solver(new_density.data(), density.data(), a, 1.0 + 4.0 * a, BoundaryHandleEnum::HandleRho);

    std::swap(density, new_density);
}

void Fluid::diffues_velocity(double dt)
{
    double a = visc * dt * (1.0 / h) * (1.0 / h);

    laplace_eq_GS_solver(new_u.data(), u.data(), a, 1.0 + 4.0 * a, BoundaryHandleEnum::HandleU);
    laplace_eq_GS_solver(new_v.data(), v.data(), a, 1.0 + 4.0 * a, BoundaryHandleEnum::HandleV);

    std::swap(new_u, u);
    std::swap(new_v, v);
}

void Fluid::project()
{
    /* This function will remove divergence component from velocity field */

    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= N; j++)
        {
            divergence[at(i, j)] = -0.5 * h * (u[at(i + 1, j)] - u[at(i - 1, j)] + v[at(i, j + 1)] - v[at(i, j - 1)]);
            p[at(i, j)] = 0.0;
        }
    }

    handle_boundaries(BoundaryHandleEnum::HandleRho, divergence.data());
    handle_boundaries(BoundaryHandleEnum::HandleRho, p.data());

    /* Now we want to find such p's that they equal divergence */
    laplace_eq_GS_solver(p.data(), divergence.data(), 1.0, 4.0, BoundaryHandleEnum::HandleRho);

    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= N; j++)
        {
            u[at(i, j)] -= 0.5 * (p[at(i + 1, j)] - p[at(i - 1, j)]) / h;
            v[at(i, j)] -= 0.5 * (p[at(i, j + 1)] - p[at(i, j - 1)]) / h;
        }
    }

    handle_boundaries(BoundaryHandleEnum::HandleU, u.data());
    handle_boundaries(BoundaryHandleEnum::HandleV, v.data());
}

void Fluid::advect_field(double dt, double *input_field, double *output_field)
{
    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= N; j++)
        {
            double x = i * h - dt * N * u[at(i, j)];
            double y = j * h - dt * N * v[at(i, j)];

            /* Field sampling already clamps point to simulation domain */
            output_field[at(i, j)] = sample_field(x, y, input_field);
        }
    }
}

void Fluid::advect_density(double dt)
{
    advect_field(dt, density.data(), new_density.data());
    handle_boundaries(BoundaryHandleEnum::HandleRho, new_density.data());
    std::swap(density, new_density);
}

void Fluid::advect_velocity(double dt)
{
    advect_field(dt, u.data(), new_u.data());
    advect_field(dt, v.data(), new_v.data());

    handle_boundaries(BoundaryHandleEnum::HandleU, new_u.data());
    handle_boundaries(BoundaryHandleEnum::HandleV, new_v.data());

    std::swap(new_u, u);
    std::swap(new_v, v);
}

void Fluid::handle_boundaries(enum BoundaryHandleEnum e, double *data)
{
    switch (e)
    {
    case BoundaryHandleEnum::HandleU:
        for (int y = 1; y <= N; y++)
        {
            data[at(0, y)] = -data[at(1, y)];
            data[at(N + 1, y)] = -data[at(N, y)];
        }
        for (int x = 1; x <= N; x++)
        {
            data[at(x, 0)] = data[at(x, 1)];
            data[at(x, N + 1)] = data[at(x, N)];
        }
        break;
    case BoundaryHandleEnum::HandleV:
        for (int y = 1; y <= N; y++)
        {
            data[at(0, y)] = data[at(1, y)];
            data[at(N + 1, y)] = data[at(N, y)];
        }
        for (int x = 1; x <= N; x++)
        {
            data[at(x, 0)] = -data[at(x, 1)];
            data[at(x, N + 1)] = -data[at(x, N)];
        }
        break;
    case BoundaryHandleEnum::HandleRho:
        for (int x = 1; x <= N; x++)
        {
            data[at(x, 0)] = data[at(x, 1)];
            data[at(x, N + 1)] = data[at(x, N)];
        }
        for (int y = 1; y <= N; y++)
        {
            data[at(0, y)] = data[at(1, y)];
            data[at(N + 1, y)] = data[at(N, y)];
        }
        break;
    default:
        break;
    }

    data[at(0, 0)] = (data[at(1, 0)] + data[at(0, 1)]) / 2.0;
    data[at(N + 1, 0)] = (data[at(N, 0)] + data[at(N + 1, 1)]) / 2.0;
    data[at(0, N + 1)] = (data[at(0, N)] + data[at(1, N + 1)]) / 2.0;
    data[at(N + 1, N + 1)] = (data[at(N, N + 1)] + data[at(N + 1, N)]) / 2.0;
}

double Fluid::sample_field(double x, double y, double *field)
{
    // clamp coordinates to the interior cell-centers range: [h/2, N*h - h/2]
    double halfh = 0.5 * h;
    double xmax = N * h - halfh;
    double ymax = N * h - halfh;

    if (x < halfh)
        x = halfh;
    if (x > xmax)
        x = xmax;
    if (y < halfh)
        y = halfh;
    if (y > ymax)
        y = ymax;

    // convert to grid coordinates (1..N in cell indices, but 0..N+1 in array)
    double gx = x / h;
    double gy = y / h;

    int i0 = (int)floor(gx); // base index
    int j0 = (int)floor(gy);
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    // fractional parts
    double sx = gx - i0;
    double sy = gy - j0;

    // clamp indices to valid array bounds [0 .. N+1]
    i0 = std::clamp(i0, 0, (int)N + 1);
    i1 = std::clamp(i1, 0, (int)N + 1);
    j0 = std::clamp(j0, 0, (int)N + 1);
    j1 = std::clamp(j1, 0, (int)N + 1);

    // bilinear interpolation
    double f00 = field[at(i0, j0)];
    double f01 = field[at(i0, j1)];
    double f10 = field[at(i1, j0)];
    double f11 = field[at(i1, j1)];

    double res =
        (1.0 - sx) * (1.0 - sy) * f00 +
        (1.0 - sx) * sy * f01 +
        sx * (1.0 - sy) * f10 +
        sx * sy * f11;

    return res;
}

void Fluid::laplace_eq_GS_solver(double *x, double *b, double a, double c, enum BoundaryHandleEnum e)
{
    for (int iter = 0; iter < 20; iter++)
    {
        for (int i = 1; i <= N; i++)
        {
            for (int j = 1; j <= N; j++)
            {
                x[at(i, j)] = (b[at(i, j)] + a * (x[at(i - 1, j)] + x[at(i + 1, j)] + x[at(i, j - 1)] + x[at(i, j + 1)])) / c;
            }
        }
        handle_boundaries(e, x);
    }
}
