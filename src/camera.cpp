#include "camera.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtx/transform.hpp>

auto Camera::camera_matrix() const -> glm::mat4
{
  //  glm::vec3 direction;
  //  direction.x = std::cos(yaw_) * std::cos(pitch_);
  //  direction.y = std::sin(pitch_);
  //  direction.z = std::sin(yaw_) * std::cos(pitch_);

  constexpr glm::vec3 RIGHT = glm::vec3(1, 0, 0);
  constexpr glm::vec3 UP = glm::vec3(0, 1, 0);
  constexpr glm::vec3 FORWARD = glm::vec3(0, 0, 1);

  return glm::translate(glm::identity<glm::mat4>(), position_) *
         glm::rotate(yaw_, UP) * glm::rotate(pitch_, RIGHT) /**
         glm::rotate(pitch_, FORWARD)*/
      ;
}

auto Camera::view_matrix() const -> glm::mat4
{
  return glm::inverse(camera_matrix());
}

void Camera::move(Camera::MoveDirection direction, float speed)
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
  yaw_ += x_offset;
  pitch_ += y_offset;

  pitch_ = std::clamp(pitch_, -glm::half_pi<float>(), glm::half_pi<float>());
}