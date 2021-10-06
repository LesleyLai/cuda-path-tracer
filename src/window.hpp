#ifndef CUDA_OPENGL_BOILERPLATE_WINDOW_HPP
#define CUDA_OPENGL_BOILERPLATE_WINDOW_HPP

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <utility>

class Window {
  GLFWwindow* window_ = nullptr;

public:
  Window() = default;
  Window(int width, int height, const char* title);

  Window(const Window&) = delete;
  auto operator=(const Window&) = delete;
  Window(Window&& other) noexcept : window_{std::exchange(other.window_, {})} {}
  auto operator=(Window&& other) & noexcept -> Window&
  {
    window_ = std::exchange(other.window_, {});
    return *this;
  }

  ~Window();

  [[nodiscard]] auto get() -> GLFWwindow*
  {
    return window_;
  }

  [[nodiscard]] auto width() const -> int
  {
    int width, height;
    glfwGetWindowSize(window_, &width, &height);
    return width;
  }

  [[nodiscard]] auto height() const -> int
  {
    int width, height;
    glfwGetWindowSize(window_, &width, &height);
    return height;
  }

  void swap_buffers();

  [[nodiscard]] auto should_close() const -> bool;
  void set_should_close(bool should_close);

  void poll_events();
};

#endif // CUDA_OPENGL_BOILERPLATE_WINDOW_HPP
