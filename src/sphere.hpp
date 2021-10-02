#ifndef CUDA_PATH_TRACER_SPHERE_HPP
#define CUDA_PATH_TRACER_SPHERE_HPP

#include <optional>

#include "ray.hpp"

[[nodiscard]] __host__ __device__ auto inline ray_sphere_intersection_test(
    Ray r, glm::vec3 center, float radius, float t_out, glm::vec3 point_out,
    glm::vec3 normal_out) -> bool
{
  const auto oc = r.origin - center;

  const auto a = dot(r.direction, r.direction);
  const auto b = 2 * dot(r.direction, oc);
  const auto c = dot(oc, oc) - radius * radius;
  const auto discrimination = b * b - 4 * a * c;

  if (discrimination < 0) { return false; }

  const auto sqrt_delta = std::sqrt(discrimination);
  const auto t1 = (-b - sqrt_delta) / (2 * a);
  const auto t2 = (-b + sqrt_delta) / (2 * a);

  auto hit_record_from_t = [&](float t) {
    t_out = t;
    point_out = r(t);
    normal_out = (point_out - center) / radius;
    return true;
  };

  // Get the smaller non-negative value of t1, t2
  if (t1 >= 0) { return hit_record_from_t(t1); }
  if (t2 >= 0) { return hit_record_from_t(t2); }
  return false;
}

#endif // CUDA_PATH_TRACER_SPHERE_HPP
