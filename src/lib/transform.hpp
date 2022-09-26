#ifndef CUDA_PATH_TRACER_TRANSFORM_HPP
#define CUDA_PATH_TRACER_TRANSFORM_HPP

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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
};

[[nodiscard]] HOST_DEVICE inline auto
transform_point(const Transform& transform, const glm::vec3& point) -> glm::vec3
{
  const auto v = transform.m() * glm::vec4(point, 1.0);
  return glm::vec3(v / v.w);
}

[[nodiscard]] HOST_DEVICE inline auto
inverse_transform_ray(const Transform& transform, const Ray& ray) -> Ray
{
  const glm::vec3 origin = ray.origin + glm::vec3(transform.inverse_m()[3]);
  const glm::vec3 direction{transform.inverse_m() *
                            glm::vec4(ray.direction, 0.)};
  return Ray{origin, ray.t_min, glm::normalize(direction), ray.t_max};
}

[[nodiscard]] HOST_DEVICE inline auto
transform_normal(const Transform& transform, const glm::vec3& normal)
    -> glm::vec3
{
  const float x = normal.x;
  const float y = normal.y;
  const float z = normal.z;
  return glm::vec3{
      transform.inverse_m()[0][0] * x + transform.inverse_m()[1][0] * y +
          transform.inverse_m()[2][0] * z,
      transform.inverse_m()[0][1] * x + transform.inverse_m()[1][1] * y +
          transform.inverse_m()[2][1] * z,
      transform.inverse_m()[0][2] * x + transform.inverse_m()[1][2] * y +
          transform.inverse_m()[2][2] * z};
}

#endif // CUDA_PATH_TRACER_TRANSFORM_HPP
