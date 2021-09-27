#pragma once

#include "shader.hpp"
#include "window.hpp"

#include <chrono>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

class App {
  Window window_;
  GLuint pbo_ = 0;
  cudaGraphicsResource* pbo_cuda_resource_ = nullptr;
  ShaderProgram program_;

  GLuint position_location_ = 0;
  GLuint tex_coords_location_ = 1;
  GLuint image_ = 0;

public:
  App();
  ~App();
  App(const App&) = delete;
  auto operator=(const App&) & -> App& = delete;
  App(App&&) noexcept = delete;
  auto operator=(App&&) & noexcept -> App& = delete;

  void main_loop();

private:
  void init_VAO();

  void run_CUDA(std::chrono::system_clock::duration time_since_start);
  void render() const;
};
