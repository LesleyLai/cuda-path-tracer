#ifndef CUDA_PATH_TRACER_AABB_HPP
#define CUDA_PATH_TRACER_AABB_HPP

#include <algorithm>
#include <cfloat>
#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include "intersection.hpp"
#include "ray.hpp"

#include "cuda_utils/definitions.hpp"

// TODO: Unit test this
struct AABB {
  glm::vec3 min{FLT_MAX};
  glm::vec3 max{-FLT_MAX};

  [[nodiscard]] HOST_DEVICE auto center() const -> glm::vec3
  {
    return (min + max) / 2.0f;
  }

  [[nodiscard]] HOST_DEVICE constexpr auto is_empty() const -> bool
  {
    for (int i = 0; i < 3; ++i)
      if (min[i] > max[i]) return true;
    return false;
  }

  [[nodiscard]] HOST_DEVICE auto enclose(glm::vec3 pt) const -> AABB
  {
    return AABB{glm::min(min, pt), glm::max(max, pt)};
  }

  [[nodiscard]] HOST_DEVICE auto enclose(AABB other) const -> AABB
  {
    return AABB{glm::min(min, other.min), glm::max(max, other.max)};
  }

  [[nodiscard]] HOST_DEVICE auto extent() const -> glm::vec3
  {
    return max - min;
  }

  // @brief Returns an index to the axis with the largest extent
  [[nodiscard]] HOST_DEVICE auto max_extent() const -> int
  {
    const glm::vec3 ext = extent();
    return (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z) ? 1 : 2;
  }

  [[nodiscard]] HOST_DEVICE friend auto aabb_union(AABB lhs, AABB rhs) -> AABB
  {
    return AABB{glm::min(lhs.min, rhs.min), glm::max(lhs.max, rhs.max)};
  }

  [[nodiscard]] HOST_DEVICE auto surface_area() const -> float
  {
    const auto diagonal = max - min;

    return 2.0f * (diagonal.x * diagonal.y + diagonal.x * diagonal.z +
                   diagonal.y * diagonal.z);
  }

  // Gets the normalized offset of a point relative to the bounding box.
  // A point at the minimum point has offset `(0, 0, 0)`, and a point at the
  // maximum point has offset `(1, 1, 1)`
  [[nodiscard]] HOST_DEVICE auto offset(glm::vec3 p) const -> glm::vec3
  {
    glm::vec3 o = p - min;
    if (max.x > min.x) o.x /= max.x - min.x;
    if (max.y > min.y) o.y /= max.y - min.y;
    if (max.z > min.z) o.z /= max.z - min.z;
    return o;
  }
};

#endif // CUDA_PATH_TRACER_AABB_HPP
