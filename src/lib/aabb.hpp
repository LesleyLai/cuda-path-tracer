#ifndef CUDA_PATH_TRACER_AABB_HPP
#define CUDA_PATH_TRACER_AABB_HPP

#include <algorithm>
#include <glm/vec3.hpp>

#include "intersection.hpp"
#include "ray.hpp"

#include "cuda_utils/definitions.hpp"

struct AABB {
  glm::vec3 min{FLT_MAX};
  glm::vec3 max{-FLT_MAX};

  [[nodiscard]] HOST_DEVICE constexpr auto is_empty() const -> bool
  {
    for (int i = 0; i < 3; ++i)
      if (min[i] > max[i]) return true;
    return false;
  }
};

#endif // CUDA_PATH_TRACER_AABB_HPP
