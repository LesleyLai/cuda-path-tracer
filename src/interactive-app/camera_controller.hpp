#ifndef CUDA_PATH_TRACER_CAMERA_CONTROLLER_HPP
#define CUDA_PATH_TRACER_CAMERA_CONTROLLER_HPP

struct Camera;

class CameraController {
  Camera* camera_ = nullptr;

public:
  CameraController(Camera& camera) : camera_{&camera} {}
  virtual ~CameraController() = default;
  CameraController(const CameraController&) = delete;
  auto operator=(const CameraController&) -> CameraController& = delete;

  // Returns true if the camera moved
  [[nodiscard]] virtual auto on_key_press(int key_code) -> bool = 0;
  [[nodiscard]] virtual auto on_mouse_move(float x_offset, float y_offset)
      -> bool = 0;
  [[nodiscard]] virtual auto draw_gui() -> bool { return false; }

protected:
  [[nodiscard]] auto camera() -> Camera& { return *camera_; }
  [[nodiscard]] auto camera() const -> const Camera& { return *camera_; }
};

#endif // CUDA_PATH_TRACER_CAMERA_CONTROLLER_HPP
