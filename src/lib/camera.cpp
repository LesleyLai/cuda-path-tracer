#include "camera.hpp"

#include <glm/gtx/transform.hpp>

auto Camera::camera_matrix() const -> glm::mat4
{
  constexpr glm::vec3 right(1, 0, 0);
  constexpr glm::vec3 up(0, 1, 0);
  constexpr glm::vec3 forward(0, 0, 1);

  return glm::translate(glm::identity<glm::mat4>(), position_) *
         glm::rotate(roll_, forward) * glm::rotate(yaw_, up) *
         glm::rotate(pitch_, right);
}

void Camera::move(Camera::MoveDirection direction)
{
  const glm::vec3 in_translation = speed * [&]() {
    switch (direction) {
    case MoveDirection::up: return glm::vec3{0, 1, 0};
    case MoveDirection::down: return glm::vec3{0, -1, 0};
    case MoveDirection::left: return glm::vec3{1, 0, 0};
    case MoveDirection::right: return glm::vec3{-1, 0, 0};
    case MoveDirection::forward: return glm::vec3{0, 0, -1};
    case MoveDirection::backward: return glm::vec3{0, 0, 1};
    }
    return glm::vec3{0};
  }();

  position_ += glm::vec3(camera_matrix() * glm::vec4(in_translation, 0));
}

void Camera::mouse_move(float x_offset, float y_offset)
{
  set_yaw(yaw_ + x_offset);
  set_pitch(pitch_ + y_offset);
}

void Camera::reset()
{
  position_ = glm::vec3(0.0);
  pitch_ = 0;
  yaw_ = 0;
  roll_ = 0;
  speed = default_speed;
}

auto Camera::generate_gpu_camera() const -> GPUCamera
{
  return GPUCamera{.camera_matrix = camera_matrix(), .fov = fov()};
}
