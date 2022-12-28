#pragma once

#include "first_person_camera_controller.hpp"
#include "lib/camera.hpp"
#include "lib/configurations.hpp"
#include "lib/path_tracer.hpp"
#include "window.hpp"

#include <memory>
#include <span>

class PreviewRenderer;

class App {
  Window window_;
  std::unique_ptr<PreviewRenderer> preview_;
  PathTracer path_tracer_;

  Camera camera_;
  FirstPersonCameraController first_person_camera_controller_;

  bool is_right_clicking_ = false;
  bool hide_control_panel_ = false;

  bool enable_denoising_ = false;

  DisplayBufferType display_type_ = DisplayBufferType::final;

public:
  explicit App(const SceneDescription& scene_desc);
  ~App();
  App(const App&) = delete;
  auto operator=(const App&) & -> App& = delete;

  void main_loop();

private:
  void run_cuda();
  void draw_gui();
};
