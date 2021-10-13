#include "app.hpp"
#include "path_tracer.hpp"

#include "preview_renderer.hpp"

#include <fmt/format.h>

#include <algorithm>

#include "gui.hpp"

#include <GLFW/glfw3.h>

#include <cuda_runtime_api.h>

App::App()
{
  int gpu_device = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpu_device > device_count) {
    fmt::print(stderr, "Error: GPU device number is greater than the number of "
                       "devices! Perhaps a CUDA-capable GPU is not installed?");
    std::exit(1);
  }

  if (!glfwInit()) {
    fmt::print(stderr, "Error: Cannot initialize GLFW context");
    std::exit(1);
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  window_ = Window(800, 800, "CUDA Path Tracer");

  glfwSetWindowUserPointer(window_.get(), this);
  glfwSetFramebufferSizeCallback(
      window_.get(), [](GLFWwindow* window, int width, int height) {
        App* app = static_cast<App*>(glfwGetWindowUserPointer(window));

        app->preview_->recreate_image(width, height);
        app->path_tracer_.resize_image(width, height);

        glViewport(0, 0, width, height);
      });
  glfwSetKeyCallback(window_.get(), [](GLFWwindow* window, int key,
                                       int /*scancode*/, int action,
                                       int /*mods*/) {
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GL_TRUE);
      case GLFW_KEY_SPACE:
        app->path_tracer_.restart();
      }
    }
  });
  glfwSetErrorCallback([](int error, const char* description) {
    fmt::print(stderr, "Error {}: {}\n", error, description);
    std::fflush(stderr);
  });

  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    fmt::print(stderr, "Failed to initialize OpenGL context");
    std::exit(1);
  }

  const auto [width, height] = window_.resolution();
  preview_ = std::make_unique<PreviewRenderer>(width, height);
  path_tracer_.create_buffers(width, height);

  init_imgui(window_.get());
}

App::~App()
{
  destroy_imgui();
}

void App::run_cuda()
{
  preview_->map_pbo([&](uchar4* dev_pbo) {
    const auto resolution = window_.resolution();
    path_tracer_.path_trace(dev_pbo, resolution.width, resolution.height);
  });
}

void App::main_loop()
{
  while (!window_.should_close()) {
    window_.poll_events();
    run_cuda();
    const auto resolution = window_.resolution();
    preview_->render(resolution.width, resolution.height);
    draw_gui();
    window_.swap_buffers();
  }
}
