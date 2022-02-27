#ifndef CUDA_PATH_TRACER_TRANSFORM_HPP
#define CUDA_PATH_TRACER_TRANSFORM_HPP

#include <glm/glm.hpp>

#include "ray.hpp"

[[nodiscard]] inline HOST_DEVICE auto transform_ray(const glm::mat4& transform,
                                                    const Ray& ray) -> Ray
{
  const glm::vec3 origin = ray.origin + glm::vec3(transform[3]);
  const glm::vec3 direction{transform * glm::vec4(ray.direction, 0.)};
  return Ray{origin, ray.t_min, glm::normalize(direction), ray.t_max};
}

#endif // CUDA_PATH_TRACER_TRANSFORM_HPP
