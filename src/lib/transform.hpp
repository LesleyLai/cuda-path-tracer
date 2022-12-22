#ifndef CUDA_PATH_TRACER_TRANSFORM_HPP
#define CUDA_PATH_TRACER_TRANSFORM_HPP

#include <glm/glm.hpp>

#include "aabb.hpp"
#include "ray.hpp"

class Transform {
  glm::mat4 m_ = glm::mat4{1.0};
  glm::mat4 inverse_m_ = glm::mat4{1.0};

public:
  Transform() = default;
  HOST_DEVICE explicit Transform(const glm::mat4& m)
      : m_(m), inverse_m_(glm::inverse(m))
  {
  }

  HOST_DEVICE Transform(const glm::mat4& m, const glm::mat4& inverse_m)
      : m_(m), inverse_m_(inverse_m)
  {
  }

  [[nodiscard]] HOST_DEVICE auto m() const -> glm::mat4 { return m_; }
  [[nodiscard]] HOST_DEVICE auto inverse_m() const -> glm::mat4
  {
    return inverse_m_;
  }

  [[nodiscard]] HOST_DEVICE auto inverse() const -> Transform
  {
    return Transform{inverse_m_, m_};
  }
};

[[nodiscard]] HOST_DEVICE inline auto
transform_point(const Transform& transform, const glm::vec3& point) -> glm::vec3
{
  const auto v = transform.m() * glm::vec4(point, 1.0);
  return glm::vec3{v} / v.w;
}

[[nodiscard]] HOST_DEVICE inline auto
transform_vector(const Transform& transform, const glm::vec3& vec) -> glm::vec3
{
  const auto v = transform.m() * glm::vec4(vec, 0.0);
  return glm::vec3{v};
}

[[nodiscard]] HOST_DEVICE inline auto
inverse_transform_ray(const Transform& transform, const Ray& ray) -> Ray
{
  const glm::vec3 origin = transform_point(transform.inverse(), ray.origin);
  const glm::vec3 direction{transform.inverse_m() *
                            glm::vec4(ray.direction, 0.)};
  return Ray{origin, ray.t_min, glm::normalize(direction), ray.t_max};
}

[[nodiscard]] HOST_DEVICE inline auto
transform_normal(const Transform& transform, const glm::vec3& normal)
    -> glm::vec3
{
  return glm::vec3{glm::transpose(transform.inverse_m()) *
                   glm::vec4(normal, 0.f)};
}

// TODO: Unit test this
[[nodiscard]] HOST_DEVICE inline auto transform_aabb(const Transform& transform,
                                                     const AABB& aabb) -> AABB
{
  if (aabb.is_empty()) return aabb;

  glm::vec3 pts[8];
  pts[0].x = pts[1].x = pts[2].x = pts[3].x = aabb.min.x;
  pts[4].x = pts[5].x = pts[6].x = pts[7].x = aabb.max.x;
  pts[0].y = pts[1].y = pts[4].y = pts[5].y = aabb.min.y;
  pts[2].y = pts[3].y = pts[6].y = pts[7].y = aabb.max.y;
  pts[0].z = pts[2].z = pts[4].z = pts[6].z = aabb.min.z;
  pts[1].z = pts[3].z = pts[5].z = pts[7].z = aabb.max.z;

  const auto p0 = transform_point(transform, pts[0]);
  AABB new_aabb{p0, p0};
  for (std::size_t i = 1; i < 8; ++i) {
    new_aabb = new_aabb.enclose(transform_point(transform, pts[i]));
  }
  return new_aabb;
}

#endif // CUDA_PATH_TRACER_TRANSFORM_HPP
