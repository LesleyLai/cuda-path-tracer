#ifndef CUDA_OPENGL_BOILERPLATE_WINDOW_HPP
#define CUDA_OPENGL_BOILERPLATE_WINDOW_HPP

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <utility>

class Window {
  int width_ = 0;
  int height_ = 0;
  GLFWwindow* window_ = nullptr;

public:
  Window() = default;
  Window(int width, int height, const char* title);

  Window(const Window&) = delete;
  auto operator=(const Window&) = delete;
  Window(Window&& other) noexcept
      : width_{std::exchange(other.width_, {})},
        height_{std::exchange(other.height_, {})}, window_{std::exchange(
                                                       other.window_, {})}
  {
  }
  auto operator=(Window&& other) & noexcept -> Window&
  {
    width_ = std::exchange(other.width_, {});
    height_ = std::exchange(other.height_, {});
    window_ = std::exchange(other.window_, {});
    return *this;
  }

  ~Window();

  [[nodiscard]] auto width() const -> int;

  [[nodiscard]] auto height() const -> int;

  void swap_buffers();

  [[nodiscard]] auto should_close() const -> bool;

  void poll_events();
};

#endif // CUDA_OPENGL_BOILERPLATE_WINDOW_HPP
