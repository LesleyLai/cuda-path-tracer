#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

class Camera {
  glm::vec3 position_ = glm::vec3(0.0);
  float pitch_ = 0;
  // float row_ = 0;
  float yaw_ = 0;

  float fov_ = glm::pi<float>() / 2.f;

public:
  [[nodiscard]] auto fov() const noexcept -> float { return fov_; }

  void set_fov(float fov) noexcept { fov_ = fov; }

  [[nodiscard]] auto camera_matrix() const -> glm::mat4;
  [[nodiscard]] auto view_matrix() const -> glm::mat4;

  enum class MoveDirection { up, down, left, right, forward, backward };
  void move(MoveDirection direction, float speed);
  void mouse_move(float x_offset, float y_offset);
};
