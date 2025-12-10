#ifndef MINMAX_HPP
#define MINMAX_HPP

#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

class SurfaceReader
{
public:
    cudaSurfaceObject_t surfObj;
    int width;

    __host__ __device__ SurfaceReader(cudaSurfaceObject_t s, int w) : surfObj(s), width(w) {}

    // Read the surface at the calculated coordinates
    __device__ float operator()(int idx) const;
};

__host__ void find_minmax_surface(cudaSurfaceObject_t surf, int width, int height, float *&minmax_d);

#endif