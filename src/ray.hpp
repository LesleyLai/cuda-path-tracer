#ifndef CUDA_PATH_TRACER_RAY_HPP
#define CUDA_PATH_TRACER_RAY_HPP

struct Ray {
  glm::vec3 origin = {};
  glm::vec3 direction = {};

  [[nodiscard]] __host__ __device__ auto operator()(float t) -> glm::vec3
  {
    return origin + direction * t;
  }
};

#endif // CUDA_PATH_TRACER_RAY_HPP
