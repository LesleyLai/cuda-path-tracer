#ifndef CUDA_PATH_TRACER_FIRST_PERSON_CAMERA_CONTROLLER_HPP
#define CUDA_PATH_TRACER_FIRST_PERSON_CAMERA_CONTROLLER_HPP

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

struct Camera;

class FirstPersonCameraController {
  Camera* camera_ = nullptr;

  glm::vec3 position_ = glm::vec3(0.0);
  float pitch_ = 0.0;
  float yaw_ = 0.0;

public:
  static constexpr float default_speed = 0.1f;
  float speed = default_speed;

  explicit FirstPersonCameraController(Camera& camera) : camera_{&camera}
  {
    reset();
  }

  [[nodiscard]] auto position() const noexcept -> glm::vec3
  {
    return position_;
  }
  void set_position(const glm::vec3& position) noexcept;

  void set_pitch(float pitch) noexcept;

  void set_yaw(float yaw) noexcept;

  // Returns true if the camera moved
  [[nodiscard]] auto handle_key_input(int key_code) -> bool;

  void mouse_move(float x_offset, float y_offset);

  /// @brief Reset the reference frame of the first person camera controller by
  /// the current camera
  void reset();

private:
  void update_camera();
};

#endif // CUDA_PATH_TRACER_FIRST_PERSON_CAMERA_CONTROLLER_HPP
