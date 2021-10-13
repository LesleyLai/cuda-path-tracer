#include "window.hpp"

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <fmt/core.h>

Window::Window(int width, int height, const char* title)
    : window_{glfwCreateWindow(static_cast<int>(width),
                               static_cast<int>(height), title, nullptr,
                               nullptr)}
{
  if (!window_) {
    fmt::print(stderr, "Error: Failed to create a GLFW Window");
    std::exit(1);
  }
  glfwMakeContextCurrent(window_);
}

Window::~Window()
{
  if (window_) { glfwDestroyWindow(window_); }
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

auto Window::resolution() const -> Resolution
{
  Resolution res;
  glfwGetWindowSize(window_, &res.width, &res.height);
  return res;
}
