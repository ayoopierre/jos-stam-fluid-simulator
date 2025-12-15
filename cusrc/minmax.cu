#include "minmax.hpp"

#include <thrust/device_ptr.h>
#include <cstdio>

__device__ float SurfaceReader::operator()(int idx) const
{

    int x = idx % width;
    int y = idx / width;

    float val;
    surf2Dread(&val, surfObj, x * sizeof(float), y);
    return val;
}

__host__ void find_minmax_surface(cudaSurfaceObject_t surf, int width, int height, float *&minmax_d)
{
    int num_pixel = width * height;
    auto counting_iterator = thrust::counting_iterator(0);
    auto surf_iter = thrust::make_transform_iterator(counting_iterator, SurfaceReader(surf, width));

    auto res = thrust::minmax_element(
        thrust::device,
        surf_iter,
        surf_iter + num_pixel);
    
    cudaMalloc(&minmax_d, 2 * sizeof(float));

    thrust::copy_n(res.first, 1, thrust::device_pointer_cast(minmax_d));
    thrust::copy_n(res.second, 1, thrust::device_pointer_cast(minmax_d + 1));
};
