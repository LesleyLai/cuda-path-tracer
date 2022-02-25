#ifndef CUDA_PATH_TRACER_RAY_GEN_CUH
#define CUDA_PATH_TRACER_RAY_GEN_CUH

#include "camera.hpp"
#include "ray.hpp"
#include <glm/glm.hpp>

[[nodiscard]] __device__ auto generate_ray(const GPUCamera& camera, float x,
                                           float y) -> Ray;

#endif // CUDA_PATH_TRACER_RAY_GEN_CUH
