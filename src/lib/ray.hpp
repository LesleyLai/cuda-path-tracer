#ifndef CUDA_PATH_TRACER_RAY_HPP
#define CUDA_PATH_TRACER_RAY_HPP

struct Ray {
  glm::vec3 origin = {};
  float t_min = {};
  glm::vec3 direction = {};
  float t_max = {};

  [[nodiscard]] __host__ __device__ auto operator()(float t) -> glm::vec3
  {
    return origin + direction * t;
  }
};

static_assert(sizeof(Ray) == sizeof(float) * 8);

#endif // CUDA_PATH_TRACER_RAY_HPP
