#include "camera.hpp"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

auto Camera::to_gpu_camera(UResolution resolution) const -> GPUCamera
{
  return GPUCamera{.camera_matrix =
                       glm::translate(glm::identity<glm::mat4>(), position) *
                       glm::mat4_cast(rotation),
                   .fov = fov,
                   .width = resolution.width,
                   .height = resolution.height};
}
void FirstPersonCameraController::update_camera()
{
  camera_->position = position_;
  camera_->rotation = glm::quat_cast(glm::yawPitchRoll(yaw_, pitch_, 0.f));
}

void FirstPersonCameraController::move(
    FirstPersonCameraController::MoveDirection direction)
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

  position_ += glm::vec3(glm::yawPitchRoll(yaw_, pitch_, 0.f) *
                         glm::vec4(in_translation, 0));

  update_camera();
}

void FirstPersonCameraController::mouse_move(float x_offset, float y_offset)
{
  set_yaw(yaw_ + x_offset);
  set_pitch(pitch_ + y_offset);

  update_camera();
}

void FirstPersonCameraController::reset()
{
  position_ = glm::vec3{0.0, 0.0, 0.0};
  pitch_ = 0;
  yaw_ = 0;
  speed = default_speed;
  update_camera();
}
