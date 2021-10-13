#pragma once

#include "path_tracer.hpp"
#include "window.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <memory>

class PreviewRenderer;

class App {
  Window window_;
  std::unique_ptr<PreviewRenderer> preview_;
  PathTracer path_tracer_;

public:
  App();
  ~App();
  App(const App&) = delete;
  auto operator=(const App&) & -> App& = delete;
  App(App&&) noexcept = delete;
  auto operator=(App&&) & noexcept -> App& = delete;

  void main_loop();

private:
  void run_cuda();
  void draw_gui();
};
