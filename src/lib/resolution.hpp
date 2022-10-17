#ifndef CUDA_PATH_TRACER_RESOLUTION_HPP
#define CUDA_PATH_TRACER_RESOLUTION_HPP

struct UResolution;

struct Resolution {
  int width = 0;
  int height = 0;

  [[nodiscard]] constexpr auto to_unsigned() const -> UResolution;
};

struct UResolution {
  unsigned int width = 0;
  unsigned int height = 0;

  [[nodiscard]] constexpr auto to_signed() const -> Resolution
  {
    return Resolution{static_cast<int>(this->width),
                      static_cast<int>(this->height)};
  }
};

[[nodiscard]] constexpr auto Resolution::to_unsigned() const -> UResolution
{
  return UResolution{static_cast<unsigned int>(this->width),
                     static_cast<unsigned int>(this->height)};
}

#endif // CUDA_PATH_TRACER_RESOLUTION_HPP
