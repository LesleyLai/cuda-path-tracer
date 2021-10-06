#include "window.hpp"

#include <fmt/core.h>

Window::Window(int width, int height, const char* title)
    : window_{glfwCreateWindow(static_cast<int>(width),
                               static_cast<int>(height), title, nullptr,
                               nullptr)}
{
  if (!window_) {
    fmt::print(stderr, "Error: Failed to create a GLFW Window");
    glfwTerminate();
    std::exit(1);
  }
  glfwMakeContextCurrent(window_);
  glfwSetErrorCallback([](int error, const char* description) {
    fmt::print(stderr, "Error {}: {}\n", error, description);
    std::fflush(stderr);
  });

  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    fmt::print(stderr, "Failed to initialize OpenGL context");
    std::exit(1);
  }
}

Window::~Window()
{
  if (window_) {
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
}

void Window::swap_buffers()
{
  glfwSwapBuffers(window_);
}
auto Window::should_close() const -> bool
{
  return glfwWindowShouldClose(window_) != 0;
}

void Window::set_should_close(bool should_close)
{
  glfwSetWindowShouldClose(window_, should_close);
}

void Window::poll_events()
{
  glfwPollEvents();
}
