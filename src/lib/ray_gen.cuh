#ifndef CUDA_PATH_TRACER_RAY_GEN_CUH
#define CUDA_PATH_TRACER_RAY_GEN_CUH

#include "ray.hpp"
#include <glm/glm.hpp>

[[nodiscard]] __device__ auto generate_ray(glm::mat4 camera_matrix, float fov,
                                           unsigned int width,
                                           unsigned int height, float x,
                                           float y) -> Ray;

#endif // CUDA_PATH_TRACER_RAY_GEN_CUH
