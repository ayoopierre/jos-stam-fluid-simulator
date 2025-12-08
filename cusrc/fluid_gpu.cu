#include "fluid_gpu.hpp"

#include <cooperative_groups.h>

#define CUDA_ERR_CHECK(err)                                                    \
    if (err < 0)                                                               \
    {                                                                          \
        printf("%s:%d - %s\n", __FUNCTION__, __LINE__, cudaGetErrorName(err)); \
        exit(1);                                                               \
    }

// clang-format off

/*
Intended to use with small blocks, 128-256 threads
to maximize occupancy (big blocks - lower granularity
for SM allocation), but not to make scheduler overhead
to large (small blocks like 32 threads - 1 warp - mean
increase scheduler overhead)
*/
__global__ static void handle_rho_bounderies_device(cudaSurfaceObject_t rho_surf, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i <= N){
        float x_1_i = surf2Dread<float>(rho_surf, 1, i);
        float x_N_i = surf2Dread<float>(rho_surf, N, i);
        float y_i_1 = surf2Dread<float>(rho_surf, i, 1);
        float y_i_N = surf2Dread<float>(rho_surf, i, N);

        surf2Dwrite<float>(x_1_i, rho_surf, 0, i);
        surf2Dwrite<float>(x_N_i, rho_surf, N, i);
        surf2Dwrite<float>(y_i_1, rho_surf, i, 0);
        surf2Dwrite<float>(y_i_N, rho_surf, i, N);
    }
}

__global__ static void handle_u_bounderies_device(cudaSurfaceObject_t u_surf, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i <= N){
        float x_1_i = surf2Dread<float>(u_surf, 1, i);
        float x_N_i = surf2Dread<float>(u_surf, N, i);
        float y_i_1 = surf2Dread<float>(u_surf, i, 1);
        float y_i_N = surf2Dread<float>(u_surf, i, N);

        surf2Dwrite<float>(-x_1_i, u_surf, 0, i);
        surf2Dwrite<float>(-x_N_i, u_surf, N, i);
        surf2Dwrite<float>(y_i_1, u_surf, i, 0);
        surf2Dwrite<float>(y_i_N, u_surf, i, N);
    }
}

__global__ static void handle_v_bounderies_device(cudaSurfaceObject_t v_surf, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i <= N){
        float x_1_i = surf2Dread<float>(v_surf, 1, i);
        float x_N_i = surf2Dread<float>(v_surf, N, i);
        float y_i_1 = surf2Dread<float>(v_surf, i, 1);
        float y_i_N = surf2Dread<float>(v_surf, i, N);

        surf2Dwrite<float>(x_1_i, v_surf, 0, i);
        surf2Dwrite<float>(x_N_i, v_surf, N, i);
        surf2Dwrite<float>(-y_i_1, v_surf, i, 0);
        surf2Dwrite<float>(-y_i_N, v_surf, i, N);
    }
}

/*
Intended to use with square tiles of threads and biggest
possible thread blocks - this uses surfaces for better 
caching (2D spacial locality) and to use shared memory
as much as possible, threads will gather memory from 
surface to shared memory (into L1) cache and the use those
for computation.
*/
__global__ static void laplace_eq_solver_step_device(
    cudaSurfaceObject_t x_next_surf,
    cudaSurfaceObject_t x_surf,
    cudaSurfaceObject_t b_surf,
    float a, float c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    /*
    We have shmem (threads.x + 2) * (threads.y + 2)
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

__global__ static void prepare_projection_surfaces_device(
    cudaSurfaceObject_t u_surf,
    cudaSurfaceObject_t v_surf,
    cudaSurfaceObject_t divergence_surf,
    cudaSurfaceObject_t p_surf, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    /* TODO: Add similr caching as in the Jacobi solver */

    if(i <= N && j <= N){
        float u_i_p1 = surf2Dread<float>(u_surf, i + 1, j);
        float u_i_m1 = surf2Dread<float>(u_surf, i - 1, j);
        float v_j_p1 = surf2Dread<float>(v_surf, i, j + 1);
        float v_j_m1 = surf2Dread<float>(v_surf, i, j - 1);

        float val = -0.5f * (u_i_p1 - u_i_m1 + v_j_p1 - v_j_m1);
        surf2Dwrite<float>(val, divergence_surf, i, j);

        surf2Dwrite<float>(0.0f, p_surf, i, j);
    }
}

__global__ static void apply_projection_surface_device(
    cudaSurfaceObject_t u_surf,
    cudaSurfaceObject_t v_surf,
    cudaSurfaceObject_t p_surf,
    cudaSurfaceObject_t x_surf,
    cudaSurfaceObject_t x_new_surf,
    int N, float h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    /* TODO: Add caching */

    if(i <= N && j <= N){
        float p_i_p1 = surf2Dread<float>(p_surf, i + 1, j);
        float p_i_m1 = surf2Dread<float>(p_surf, i - 1, j);
        float p_j_p1 = surf2Dread<float>(p_surf, i, j + 1);
        float p_j_m1 = surf2Dread<float>(p_surf, i, j - 1);

        float u = surf2Dread<float>(u_surf, i, j);
        float v = surf2Dread<float>(v_surf, i, j);

        u = u - 0.5 * (p_i_p1 - p_i_m1) / h;
        v = v - 0.5 * (p_j_p1 - p_j_m1) / h;

        surf2Dwrite<float>(u, x_surf, i, j);
        surf2Dwrite<float>(v, x_new_surf, i, j);
    }
}

__device__ static int clamp_device(int x, int min, int max)
{
    if(x < min) return min;
    if(x > max) return max;
    return x;
}

__device__ static float bilerp_device(
    cudaSurfaceObject_t src_surf,
    float x, float y,
    int N, float h)
{
    float halfh = 0.5f * h;
    float xmax = N * h - halfh;
    float ymax = N * h - halfh;

    if (x < halfh)
        x = halfh;
    if (x > xmax)
        x = xmax;
    if (y < halfh)
        y = halfh;
    if (y > ymax)
        y = ymax;

    float gx = x / h;
    float gy = y / h;

    int i0 = (int)floor(gx); // base index
    int j0 = (int)floor(gy);
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    // fractional parts
    float sx = gx - i0;
    float sy = gy - j0;

    i0 = clamp_device(i0, 0, (int)N + 1);
    i1 = clamp_device(i1, 0, (int)N + 1);
    j0 = clamp_device(j0, 0, (int)N + 1);
    j1 = clamp_device(j1, 0, (int)N + 1);

    float f00 = surf2Dread<float>(src_surf, i0, j0);
    float f01 = surf2Dread<float>(src_surf, i0, j1);
    float f10 = surf2Dread<float>(src_surf, i1, j0);
    float f11 = surf2Dread<float>(src_surf, i1, j1);

    float res =
        (1.0 - sx) * (1.0 - sy) * f00 +
        (1.0 - sx) * sy * f01 +
        sx * (1.0 - sy) * f10 +
        sx * sy * f11;
    
    return res;
}

/* 
Not much we can use here as for shared memory in this case.
We can cannot predict memory location to sample from so we
cannot cache this in shared memory. There is not much to 
leverage from surface locality as well.
*/
__global__ static void advect_field_device(
    cudaSurfaceObject_t u_surf,
    cudaSurfaceObject_t v_surf,
    cudaSurfaceObject_t source_surf,
    cudaSurfaceObject_t dest_surf,
    int N, float h, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i <= N && j <= N){
        float u = surf2Dread<float>(u_surf, i, j);
        float v = surf2Dread<float>(v_surf, i, j);

        float x = i * h - dt * N * u;
        float y = j * h - dt * N * v;

        float val = bilerp_device(source_surf, x, y, N, h);

        surf2Dwrite<float>(val, dest_surf, i * sizeof(float), j, cudaSurfaceBoundaryMode::cudaBoundaryModeZero);
    }
}
// clang-format on

FluidGpu::FluidGpu(size_t N)
{
    this->N = N;
    h = box_length / N;

    cudaError_t err;

    channel_desc = cudaCreateChannelDesc<float>();

    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    prepare_surface(&density_desc, &density_surf, &density_arr);
    prepare_surface(&u_desc, &u_surf, &u_arr);
    prepare_surface(&v_desc, &v_surf, &v_arr);

    prepare_surface(&x_desc, &x_surf, &x_arr);
    prepare_surface(&x_new_desc, &x_new_surf, &x_new_arr);

    prepare_surface(&divergence_desc, &divergence_surf, &divergence_arr);
    prepare_surface(&p_desc, &p_surf, &p_arr);
}

void FluidGpu::diffuse_density(float dt)
{
    float a = diff * dt * (1.0 / h) * (1.0 / h);

    solve_laplace_eq_JA_solver(x_new_surf, x_surf, density_surf, a, 1.0 + 4.0 * a, BoundaryHandleEnum::HandleRho);

    /* New density surface is stored in coresponding x surface*/
    std::swap(x_surf, density_surf);
}

void FluidGpu::diffuse_velocity(float dt)
{
    float a = visc * dt * (1.0 / h) * (1.0 / h);

    solve_laplace_eq_JA_solver(x_new_surf, x_surf, u_surf, a, 1.0 + 4.0 * a, BoundaryHandleEnum::HandleRho);
    /* New u surface is stored in coresponding x surface*/
    std::swap(x_surf, u_surf);

    solve_laplace_eq_JA_solver(x_new_surf, x_surf, v_surf, a, 1.0 + 4.0 * a, BoundaryHandleEnum::HandleRho);
    /* New u surface is stored in coresponding x surface*/
    std::swap(x_surf, v_surf);
}

void FluidGpu::project(float dt)
{
    cudaError_t err;
    int blocks_per_axis = (N / 32) + (N % 32 ? 1 : 0);
    dim3 blocks(blocks_per_axis, blocks_per_axis);

    int mod = ((N + 2) * (N + 2)) % 128;
    int blc_bd_task = ((N + 2) * (N + 2)) / 128 + (mod != 0 ? 1 : 0);
    dim3 blocks_for_bd_task(blc_bd_task);

    prepare_projection_surfaces_device<<<blocks, dim3(32, 32)>>>(
        u_surf, v_surf,
        divergence_surf,
        p_surf, N);

    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    /* Boundary condition */
    handle_rho_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(divergence_surf, N);
    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    handle_rho_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(p_surf, N);
    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    solve_laplace_eq_JA_solver(x_surf, divergence_surf, p_surf, 1.0, 4.0, BoundaryHandleEnum::HandleRho);

    std::swap(p_surf, x_surf);

    apply_projection_surface_device<<<blocks, dim3(32, 32)>>>(
        u_surf, v_surf, p_surf,
        x_surf, x_new_surf, N, h);

    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    std::swap(u_surf, x_surf);
    std::swap(v_surf, x_new_surf);

    /* Handle boundary conditions */
    handle_u_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(u_surf, N);
    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    handle_v_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(v_surf, N);
    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);
}

void FluidGpu::advect_density(float dt)
{
    cudaError_t err;
    int blocks_per_axis = (N / 32) + (N % 32 ? 1 : 0);
    dim3 blocks(blocks_per_axis, blocks_per_axis);

    int mod = ((N + 2) * (N + 2)) % 128;
    int blc_bd_task = ((N + 2) * (N + 2)) / 128 + (mod != 0 ? 1 : 0);
    dim3 blocks_for_bd_task(blc_bd_task);

    advect_field_device<<<blocks, dim3(32, 32)>>>(
        u_surf, v_surf, density_surf,
        x_surf, N, h, dt);

    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    std::swap(x_surf, density_surf);

    /* !!! HANDLE BOUNDARY CONDITION !!! */
    handle_rho_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(density_surf, N);
    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);
}

void FluidGpu::advect_velocity(float dt)
{
    cudaError_t err;
    int blocks_per_axis = (N / 32) + (N % 32 ? 1 : 0);
    dim3 blocks(blocks_per_axis, blocks_per_axis);

    int mod = ((N + 2) * (N + 2)) % 128;
    int blc_bd_task = ((N + 2) * (N + 2)) / 128 + (mod != 0 ? 1 : 0);
    dim3 blocks_for_bd_task(blc_bd_task);

    advect_field_device<<<blocks, dim3(32, 32)>>>(u_surf, v_surf, u_surf, x_surf, N, h, dt);

    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    advect_field_device<<<blocks, dim3(32, 32)>>>(
        u_surf, v_surf, v_surf,
        x_new_surf, N, h, dt);

    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    std::swap(x_surf, u_surf);
    std::swap(x_new_surf, v_surf);

    /* !!! HANDLE BOUNDARY CONDITION !!! */

    /* U boundary condition */
    handle_u_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(u_surf, N);
    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);

    /* V boundary condition */
    handle_v_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(v_surf, N);
    err = cudaGetLastError();
    CUDA_ERR_CHECK(err);
}

void FluidGpu::solve_laplace_eq_JA_solver(cudaSurfaceObject_t x_new, cudaSurfaceObject_t x, cudaSurfaceObject_t b, float a, float c, enum BoundaryHandleEnum e)
{
    cudaError_t err;
    int blocks_per_axis = (N / 32) + (N % 32 ? 1 : 0);
    dim3 blocks(blocks_per_axis, blocks_per_axis);

    int mod = ((N + 2) * (N + 2)) % 128;
    int blc_bd_task = ((N + 2) * (N + 2)) / 128 + (mod != 0 ? 1 : 0);
    dim3 blocks_for_bd_task(blc_bd_task);

    for (int iter = 0; iter < 20; iter++)
    {
        laplace_eq_solver_step_device<<<blocks, dim3(32, 32), sizeof(float) * 32 * 32>>>(
            x_new, x, b, a, c, N);

        err = cudaGetLastError();
        CUDA_ERR_CHECK(err);

        switch (e)
        {
        case BoundaryHandleEnum::HandleRho:
            handle_rho_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(x_new, N);
            break;
        case BoundaryHandleEnum::HandleU:
            handle_u_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(x_new, N);
            break;
        case BoundaryHandleEnum::HandleV:
            handle_v_bounderies_device<<<blocks_for_bd_task, dim3(128)>>>(x_new, N);
            break;
        default:
            break;
        }

        err = cudaGetLastError();
        CUDA_ERR_CHECK(err);

        std::swap(x_new, x);
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
