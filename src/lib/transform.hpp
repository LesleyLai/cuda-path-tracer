#ifndef CUDA_PATH_TRACER_TRANSFORM_HPP
#define CUDA_PATH_TRACER_TRANSFORM_HPP

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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

[[nodiscard]] HOST_DEVICE inline auto transform_aabb(const Transform& transform,
                                                     const AABB& aabb) -> AABB
{
  if (aabb.is_empty()) return aabb;

  const auto p1 = transform_point(transform, aabb.min);
  const auto p2 = transform_point(transform, aabb.max);

  return AABB{glm::min(p1, p2), glm::max(p1, p2)};
}

#endif // CUDA_PATH_TRACER_TRANSFORM_HPP
