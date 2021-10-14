#include "camera.hpp"

auto Camera::camera_matrix() const -> glm::mat4
{
  return glm::translate(glm::identity<glm::mat4>(), position_);
}

void Camera::move(Camera::MoveDirection direction, float speed)
{
  const glm::vec3 in_translation = speed * [&]() {
    switch (direction) {
    case MoveDirection::up: return glm::vec3{0, 1, 0};
    case MoveDirection::down: return glm::vec3{0, -1, 0};
    case MoveDirection::left: return glm::vec3{1, 0, 0};
    case MoveDirection::right: return glm::vec3{-1, 0, 0};
    }
  }();

  position_ += glm::vec3(camera_matrix() * glm::vec4(in_translation, 0));
}
