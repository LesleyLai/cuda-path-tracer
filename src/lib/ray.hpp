#ifndef CUDA_PATH_TRACER_RAY_HPP
#define CUDA_PATH_TRACER_RAY_HPP

#include <glm/vec3.hpp>

#include "cuda_utils/definitions.hpp"

struct Ray {
  glm::vec3 origin = {};
  float t_min = {};
  glm::vec3 direction = {};
  float t_max = {};

  [[nodiscard]] HOST_DEVICE auto operator()(float t) const -> glm::vec3
  {
    return origin + direction * t;
  }
};

static_assert(sizeof(Ray) == sizeof(float) * 8);

#endif // CUDA_PATH_TRACER_RAY_HPP
