#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

class Camera {
  glm::vec3 position_ = glm::vec3(0.0);

private:
  float pitch_ = 0;
  // float roll_ = 0;
  float yaw_ = 0;

  float fov_ = glm::pi<float>() / 2.f;

public:
  static constexpr float default_speed = 0.01;
  float speed = default_speed;

  [[nodiscard]] auto position() const noexcept -> glm::vec3
  {
    return position_;
  }
  void set_position(const glm::vec3& position) noexcept
  {
    position_ = position;
  }
  [[nodiscard]] auto pitch() const noexcept -> float { return pitch_; }
  void set_pitch(float pitch) noexcept
  {
    pitch_ = glm::clamp(pitch, -glm::half_pi<float>(), glm::half_pi<float>());
  }
  [[nodiscard]] auto yaw() const noexcept -> float { return yaw_; }
  void set_yaw(float yaw) noexcept
  {
    while (yaw > glm::pi<float>())
      yaw -= glm::two_pi<float>();
    while (yaw < -glm::pi<float>())
      yaw += glm::two_pi<float>();
    yaw_ = yaw;
  }
  [[nodiscard]] auto fov() const noexcept -> float { return fov_; }
  void set_fov(float fov) noexcept { fov_ = fov; }

  [[nodiscard]] auto camera_matrix() const -> glm::mat4;

  enum class MoveDirection { up, down, left, right, forward, backward };
  void move(MoveDirection direction);
  void mouse_move(float x_offset, float y_offset);

  /// @brief Reset camera states to default
  void reset();
};
