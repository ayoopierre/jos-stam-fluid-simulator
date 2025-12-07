#ifndef FLUID_HPP
#define FLUID_HPP

#define GS_ITER 30

#include <cuda_runtime.h>
#include <cuda.h>

#include <vector>
#include <cmath>
#include <cstdio>

#include "fluid.hpp"
#include "utils.hpp"

class Window;

class FluidGpu : public Fluid
{
public:
    FluidGpu(size_t N);

    inline void step(float dt)
    {
    }

private:
    void solve_laplace_eq_JA_solver(cudaSurfaceObject_t x_new,
                                    cudaSurfaceObject_t x, cudaSurfaceObject_t b,
                                    float a, float c);
    void prepare_surface(cudaResourceDesc *desc, cudaSurfaceObject_t *surf, cudaArray_t *arr);

    /* */
    cudaChannelFormatDesc channel_desc;

    /* Resource handles */
    cudaResourceDesc density_desc;
    cudaResourceDesc u_desc;
    cudaResourceDesc v_desc;

    cudaResourceDesc temp_desc;

    cudaResourceDesc divergence_desc;
    cudaResourceDesc p_desc;

    /* Surface handles */
    cudaSurfaceObject_t density_surf;
    cudaSurfaceObject_t u_surf;
    cudaSurfaceObject_t v_surf;

    cudaSurfaceObject_t temp_surf;

    cudaSurfaceObject_t divergence_surf;
    cudaSurfaceObject_t p_surf;

    /* Array handles */
    cudaArray_t *density_arr;
    cudaArray_t *u_arr;
    cudaArray_t *v_arr;

    cudaArray_t *temp_arr;

    cudaArray_t *divergence_arr;
    cudaArray_t *p_arr;

    /* Fluid params */
    size_t N;

    float box_length = 1.0;
    float h;

    float diff = 1.01;
    float visc = 0.01;
};

#endif