#ifndef CUDA_PATH_TRACER_CUDA_BUFFER_HPP
#define CUDA_PATH_TRACER_CUDA_BUFFER_HPP

#include <cstdlib>
#include <utility>

#include "cuda_check.hpp"
#include "span.hpp"

namespace cuda {

template <typename T> class Buffer {
  T* ptr_ = nullptr;

public:
  Buffer() = default;
  /* explicit(false) */ Buffer(T* ptr) : ptr_{ptr} {}
  /* explicit(false) */ Buffer(std::nullptr_t) {}

  ~Buffer() { cudaFree(ptr_); }

  Buffer(const Buffer&) = delete;
  auto operator=(const Buffer&) & -> Buffer& = delete;
  Buffer(Buffer&& rhs) noexcept : ptr_(std::exchange(rhs.ptr_, nullptr)) {}

  auto operator=(Buffer&& rhs) & noexcept -> Buffer&
  {
    if (this != &rhs) { ptr_ = std::exchange(rhs.ptr_, nullptr); }
    return *this;
  }

  [[nodiscard]] auto data() -> T* { return ptr_; }

  [[nodiscard]] auto data() const -> const T* { return ptr_; }
};

template <typename T>
[[nodiscard]] auto make_buffer(std::size_t size) -> Buffer<T>
{
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), size * sizeof(T)));
  return Buffer{ptr};
}

template <typename T>
[[nodiscard]] auto create_buffer_from_cpu_data(Span<const T> span)
{
  auto dev_buffer = cuda::make_buffer<T>(span.size());
  CUDA_CHECK(cudaMemcpy(dev_buffer.data(), span.data(), span.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return dev_buffer;
}

} // namespace cuda

#endif // CUDA_PATH_TRACER_CUDA_BUFFER_HPP
