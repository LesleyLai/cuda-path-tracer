#include "ray_gen.cuh"

[[nodiscard]] __device__ auto generate_ray(const GPUCamera& camera, float x,
                                           float y) -> Ray
{
  const float aspect_ratio =
      static_cast<float>(camera.width) / static_cast<float>(camera.height);

  const float viewport_height = 2.0f * tan(camera.fov / 2);
  const float viewport_width = aspect_ratio * viewport_height;
  const float focal_length = 1.0;

  const auto origin = glm::vec3(0, 0, 0);
  const auto horizontal = glm::vec3(viewport_width, 0, 0);
  const auto vertical = glm::vec3(0, viewport_height, 0);
  const auto lower_left_corner = origin - horizontal / 2.f - vertical / 2.f -
                                 glm::vec3(0, 0, focal_length);

  const auto u = x / static_cast<float>(camera.width - 1);
  const auto v = y / static_cast<float>(camera.height - 1);
  const auto direction =
      lower_left_corner + u * horizontal + v * vertical - origin;

  const auto world_origin =
      glm::vec3(camera.camera_matrix * glm::vec4(origin, 1.0));
  const auto world_direction = glm::normalize(
      glm::vec3(camera.camera_matrix * glm::vec4(direction, 0.0)));
  return Ray{world_origin, 1e-4, world_direction, FLT_MAX};
}
