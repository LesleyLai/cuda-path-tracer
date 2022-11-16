#ifndef CUDA_PATH_TRACER_RAY_GEN_CUH
#define CUDA_PATH_TRACER_RAY_GEN_CUH

#include "camera.hpp"
#include "ray.hpp"
#include "span.hpp"

[[nodiscard]] __device__ auto generate_ray(const GPUCamera& camera, float x,
                                           float y) -> Ray;

void generate_rays(unsigned int iteration, const Camera& camera,
                   UResolution resolution, Ray* rays, int* pixel_indices);

#endif // CUDA_PATH_TRACER_RAY_GEN_CUH
