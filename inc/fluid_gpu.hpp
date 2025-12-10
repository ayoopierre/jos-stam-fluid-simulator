#ifndef FLUID_GPU_HPP
#define FLUID_GPU_HPP

#define GS_ITER 30

#include <cuda_runtime.h>
#include <cuda.h>

#include <vector>
#include <cmath>
#include <cstdio>

#include "fluid.hpp"
#include "utils.hpp"

#define CUDA_ERR_CHECK(err)                                                    \
    if (err != 0)                                                              \
    {                                                                          \
        printf("%s:%d - %s\n", __FUNCTION__, __LINE__, cudaGetErrorName(err)); \
        exit(1);                                                               \
    }

class Window;

class FluidGpu : public Fluid
{
public:
    __host__ FluidGpu(size_t N);

    __host__ inline void step(float dt)
    {
        apply_sources(dt);
        diffuse_density(dt);
        advect_density(dt);

        diffuse_velocity(dt);
        project(dt);
        advect_velocity(dt);
        project(dt);

        cudaError_t err;
        err = cudaDeviceSynchronize();
        CUDA_ERR_CHECK(err);
    }

    __host__ bool draw_into_bitmap(MyBitmap &bitmap);

private:
    enum class BoundaryHandleEnum
    {
        HandleU,
        HandleV,
        HandleRho
    };

    __host__ void apply_sources(float dt);
    __host__ void diffuse_density(float dt);
    __host__ void diffuse_velocity(float dt);
    __host__ void project(float dt);
    __host__ void advect_density(float dt);
    __host__ void advect_velocity(float dt);

    __host__ void solve_laplace_eq_JA_solver(cudaSurfaceObject_t x_new,
                                             cudaSurfaceObject_t x, cudaSurfaceObject_t b,
                                             float a, float c, enum BoundaryHandleEnum e);
    __host__ void prepare_surface(cudaResourceDesc *desc, cudaSurfaceObject_t *surf, cudaArray_t *arr);

    /* */
    cudaChannelFormatDesc channel_desc;
    cudaGraphicsResource_t cuda_graphics_resource;

    /* Resource handles */
    cudaResourceDesc density_desc;
    cudaResourceDesc u_desc;
    cudaResourceDesc v_desc;

    cudaResourceDesc x_desc;
    cudaResourceDesc x_new_desc;

    cudaResourceDesc divergence_desc;
    cudaResourceDesc p_desc;

    /* Surface handles */
    cudaSurfaceObject_t density_surf;
    cudaSurfaceObject_t u_surf;
    cudaSurfaceObject_t v_surf;

    cudaSurfaceObject_t x_surf;
    cudaSurfaceObject_t x_new_surf;

    cudaSurfaceObject_t divergence_surf;
    cudaSurfaceObject_t p_surf;

    /* Array handles */
    cudaArray_t density_arr;
    cudaArray_t u_arr;
    cudaArray_t v_arr;

    cudaArray_t x_arr;
    cudaArray_t x_new_arr;

    cudaArray_t divergence_arr;
    cudaArray_t p_arr;

    /* Fluid params */
    size_t N;

    float box_length = 1.0;
    float h;

    float diff = 1.01;
    float visc = 0.01;
};

#endif