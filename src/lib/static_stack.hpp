#ifndef CUDA_PATH_TRACER_STATIC_STACK_HPP
#define CUDA_PATH_TRACER_STATIC_STACK_HPP

#include <cassert>

template <typename T, unsigned int N> class StaticStack {
public:
  T data_[N];
  unsigned int size_ = 0;

  constexpr StaticStack() noexcept = default;

  [[nodiscard]] constexpr auto size() noexcept -> unsigned int { return size_; }

  [[nodiscard]] constexpr auto top() noexcept -> T
  {
    assert(size_ != 0);
    return data_[size_ - 1];
  }

  constexpr void push(T value) noexcept
  {
    assert(size_ < N);
    data_[size_++] = value;
  }

  constexpr auto pop() noexcept -> T
  {
    assert(size_ != 0);
    T value = top();
    --size_;
    return value;
  }
};

#endif // CUDA_PATH_TRACER_STATIC_STACK_HPP
