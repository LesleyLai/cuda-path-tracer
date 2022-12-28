#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

#include "resolution.hpp"
#include <fmt/format.h>

struct GPUCamera {
  glm::mat4 camera_matrix = {};
  float vfov = 0;
  unsigned int width = 0;
  unsigned int height = 0;
};

struct Camera {
  glm::vec3 position{};
  glm::quat rotation = {1.0, 0.0, 0.0, 0.0};
  float vfov = glm::pi<float>() / 2.f;

  [[nodiscard]] auto to_gpu_camera(UResolution resolution) const -> GPUCamera;
};
