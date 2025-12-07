#include "fluid_gpu.hpp"

#include <cooperative_groups.h>

#define CUDA_ERR_CHECK(err)                                                    \
    if (err < 0)                                                               \
    {                                                                          \
        printf("%s:%d - %s\n", __FUNCTION__, __LINE__, cudaGetErrorName(err)); \
        exit(1);                                                               \
    }

// clang-format off
__global__ static void handle_bounderies_device(cudaSurfaceObject_t x_surf, int N)
{

}

__global__ static void laplace_eq_solver_step_device(
    cudaSurfaceObject_t x_next_surf,
    cudaSurfaceObject_t x_surf,
    cudaSurfaceObject_t b_surf,
    float a, float c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    /*
    We have block (threads.x + 2) * (threads.y + 2)
    and since we got 1024 threads per group we have
    8Kb of shared memory for tile caching which should
    fit in any GPU L1 cache.
    */
    extern __shared__ float tile[];
    int w = blockDim.x + 2;
    int local_idx = (threadIdx.y + 1) * w + (threadIdx.x);

    /* Each thread within group will load into tile */
    tile[local_idx] = surf2Dread<float>(x_surf, i, j);

    if(threadIdx.x == 0)
        tile[local_idx - 1] = surf2Dread<float>(x_surf, i - 1, j);
    if(threadIdx.x == blockDim.x - 1)
        tile[local_idx + 1] = surf2Dread<float>(x_surf, i + 1, j);
    if(threadIdx.y == 0)
        tile[local_idx - w] = surf2Dread<float>(x_surf, i, j - 1);
    if(threadIdx.y == blockDim.y - 1)
        tile[local_idx + w] = surf2Dread<float>(x_surf, i, j + 1);

    __syncthreads();

    if(i <= N && j <= N){
        float s = tile[local_idx - 1] + tile[local_idx + 1] +
            tile[local_idx + w] + tile[local_idx - w];
        
        float b = surf2Dread<float>(b_surf, i, j);

        float val = (b + a * s) / c;

        surf2Dwrite<float>(val, x_next_surf, i * sizeof(float), j, cudaSurfaceBoundaryMode::cudaBoundaryModeZero);
    }
}
// clang-format on

FluidGpu::FluidGpu(size_t N)
{
    this->N = N;
    h = box_length / N;

    cudaError_t err;

    channel_desc = cudaCreateChannelDesc<float>();

    prepare_surface(&density_desc, &density_surf, density_arr);
    prepare_surface(&u_desc, &u_surf, u_arr);
    prepare_surface(&v_desc, &v_surf, v_arr);

    prepare_surface(&temp_desc, &temp_surf, temp_arr);

    prepare_surface(&divergence_desc, &divergence_surf, divergence_arr);
    prepare_surface(&p_desc, &p_surf, p_arr);
}

void FluidGpu::solve_laplace_eq_JA_solver(cudaSurfaceObject_t x_new, cudaSurfaceObject_t x, cudaSurfaceObject_t b, float a, float c)
{
    cudaError_t err;
    int blocks_per_axis = (N / 32) + (N % 32 ? 1 : 0);
    dim3 blocks(blocks_per_axis, blocks_per_axis);

    for (int iter = 0; iter < 20; iter++)
    {
        laplace_eq_solver_step_device<<<blocks, dim3(32, 32), sizeof(float) * 32 * 32>>>(
            x_new, x, b, a, c, N
        );
    }
}

void FluidGpu::prepare_surface(cudaResourceDesc *desc, cudaSurfaceObject_t *surf, cudaArray_t *arr)
{
    cudaError_t err;

    err = cudaMallocArray(arr, &channel_desc, N + 2, N + 2, cudaArraySurfaceLoadStore);
    CUDA_ERR_CHECK(err);

    std::memset(desc, 0, sizeof(*desc));
    desc->resType = cudaResourceTypeArray;
    desc->res.array.array = *arr;

    err = cudaCreateSurfaceObject(surf, desc);
    CUDA_ERR_CHECK(err);
}
