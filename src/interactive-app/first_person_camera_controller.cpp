#include "first_person_camera_controller.hpp"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

#include <glfw/glfw3.h>

#include "lib/camera.hpp"

#include <imgui.h>

void FirstPersonCameraController::update_camera()
{
  camera().position = position_;
  camera().rotation = glm::quat_cast(glm::yawPitchRoll(yaw_, pitch_, 0.f));
}

void FirstPersonCameraController::reset()
{
  position_ = camera().position;

  const glm::vec3 eulers = glm::eulerAngles(camera().rotation);

  pitch_ = eulers.x;
  yaw_ = eulers.y;
  // fmt::print("{}\n", eulers.z);

  speed = default_speed;
  update_camera();
}

void FirstPersonCameraController::set_position(
    const glm::vec3& position) noexcept
{
  position_ = position;
  // camera_->position = position;
}

void FirstPersonCameraController::set_pitch(float pitch) noexcept
{
  pitch_ = glm::clamp(pitch, -glm::half_pi<float>(), glm::half_pi<float>());
}

void FirstPersonCameraController::set_yaw(float yaw) noexcept
{
  const auto pi = glm::pi<float>();
  const auto two_pi = glm::two_pi<float>();
  yaw = fmod(yaw + pi, two_pi);
  if (yaw < 0.0) yaw += two_pi;
  yaw_ = yaw - pi;
}

auto FirstPersonCameraController::on_key_press(int key) -> bool
{
  glm::vec3 direction;
  switch (key) {
  case GLFW_KEY_R:
    direction = glm::vec3{0, 1, 0}; // up
    break;
  case GLFW_KEY_F:
    direction = glm::vec3{0, -1, 0}; // down
    break;
  case GLFW_KEY_A:
    direction = glm::vec3{1, 0, 0}; // left
    break;
  case GLFW_KEY_D:
    direction = glm::vec3{-1, 0, 0}; // right
    break;
  case GLFW_KEY_W:
    direction = glm::vec3{0, 0, -1}; // forward
    break;
  case GLFW_KEY_S:
    direction = glm::vec3{0, 0, 1}; // backward
    break;
  default: return false;
  }

  const glm::vec3 in_translation = speed * direction;
  position_ += glm::vec3(glm::yawPitchRoll(yaw_, pitch_, 0.f) *
                         glm::vec4(in_translation, 0));

  update_camera();

  return true;
}

auto FirstPersonCameraController::on_mouse_move(float x_offset, float y_offset)
    -> bool
{
  set_yaw(yaw_ + x_offset);
  set_pitch(pitch_ + y_offset);

  update_camera();
  return true;
}

auto FirstPersonCameraController::draw_gui() -> bool
{
  bool camera_moved = false;

  ImGui::Text("w/a/s/d: forward/left/backward/right");
  ImGui::Text("r/f: up/down");
  ImGui::Text("mouse right drag: pitch/yaw");

  ImGui::NewLine();
  ImGui::Text("Transformation:");

  if (ImGui::Button("Reset")) {
    reset();
    camera_moved = true;
    update_camera();
  }

  if (glm::vec3 position = this->position();
      ImGui::InputFloat3("Translation", &position[0])) {
    set_position(position);
    camera_moved = true;
    update_camera();
  }

  ImGui::NewLine();
  ImGui::Text("Movement:");
  ImGui::SliderFloat("Speed", &speed, 0.001f, 100, "%.3f",
                     ImGuiSliderFlags_Logarithmic);
  return camera_moved;
}