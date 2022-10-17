#ifndef CUDA_OPENGL_BOILERPLATE_WINDOW_HPP
#define CUDA_OPENGL_BOILERPLATE_WINDOW_HPP

#include <utility>

struct GLFWwindow;

struct Resolution {
  int width = 0;
  int height = 0;
};

struct UResolution {
  unsigned int width = 0;
  unsigned int height = 0;
};

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

  [[nodiscard]] auto get() -> GLFWwindow* { return window_; }

  [[nodiscard]] auto resolution() const -> Resolution;
  [[nodiscard]] auto u_resolution() const -> UResolution;

  void swap_buffers();

  [[nodiscard]] auto should_close() const -> bool;
  void set_should_close(bool should_close);

  void poll_events();
};

#endif // CUDA_OPENGL_BOILERPLATE_WINDOW_HPP
