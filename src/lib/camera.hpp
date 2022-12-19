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

class FirstPersonCameraController {
  Camera* camera_ = nullptr;

  glm::vec3 position_ = glm::vec3(0.0);
  float fov_ = glm::pi<float>() / 2.f;
  float pitch_ = 0.0;
  float yaw_ = 0.0;

public:
  static constexpr float default_speed = 0.01f;
  float speed = default_speed;

  explicit FirstPersonCameraController(Camera& camera) : camera_{&camera} {}

  [[nodiscard]] auto position() const noexcept -> glm::vec3
  {
    return position_;
  }
  void set_position(const glm::vec3& position) noexcept
  {
    position_ = position;
    camera_->position = position;
  }

  void set_pitch(float pitch) noexcept
  {
    pitch_ = glm::clamp(pitch, -glm::half_pi<float>(), glm::half_pi<float>());
  }
  void set_yaw(float yaw) noexcept
  {
    const auto pi = glm::pi<float>();
    const auto two_pi = glm::two_pi<float>();
    yaw = fmod(yaw + pi, two_pi);
    if (yaw < 0.0) yaw += two_pi;
    yaw_ = yaw - pi;
  }

  [[nodiscard]] auto fov() const noexcept -> float { return fov_; }
  void set_fov(float fov) noexcept { fov_ = fov; }

  // Move and update the camera
  enum class MoveDirection { up, down, left, right, forward, backward };
  void move(MoveDirection direction);
  void mouse_move(float x_offset, float y_offset);

  /// @brief Reset the reference frame of the first person camera controller by
  /// the current camera
  void reset();

private:
  void update_camera();
};