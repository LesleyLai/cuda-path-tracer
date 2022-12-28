#include "camera.hpp"

#include <glm/gtx/transform.hpp>

auto Camera::to_gpu_camera(UResolution resolution) const -> GPUCamera
{
  return GPUCamera{.camera_matrix =
                       glm::translate(glm::identity<glm::mat4>(), position) *
                       glm::mat4_cast(rotation),
                   .vfov = vfov,
                   .width = resolution.width,
                   .height = resolution.height};
}
