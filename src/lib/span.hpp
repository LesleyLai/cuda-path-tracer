#ifndef CUDA_PATH_TRACER_SPAN_HPP
#define CUDA_PATH_TRACER_SPAN_HPP

#include <cassert>
#include <iterator>

#include "cuda_utils/definitions.hpp"

template <typename T> struct Span {
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;

private:
  pointer ptr_ = nullptr;
  size_type size_ = 0;

public:
  constexpr Span() noexcept = default;
  HOST_DEVICE constexpr Span(pointer first, size_type size) noexcept
      : ptr_{first}, size_{size}
  {
  }

  template <std::size_t N>
  HOST_DEVICE constexpr Span(element_type (&arr)[N]) noexcept
      : ptr_{arr}, size_{N}
  {
  }

  template <class U,
            class = std::enable_if_t<!std::is_same_v<T, U> &&
                                     std::is_same_v<std::remove_const_t<T>, U>>>
  HOST_DEVICE constexpr Span(Span<U> other) noexcept
      : ptr_{other.data()}, size_{other.size()}
  {
  }

  [[nodiscard]] HOST_DEVICE constexpr auto begin() const noexcept -> iterator
  {
    return ptr_;
  }

  [[nodiscard]] HOST_DEVICE constexpr auto end() const noexcept -> iterator
  {
    return ptr_ + size_;
  }

  [[nodiscard]] HOST_DEVICE constexpr auto data() const noexcept -> pointer
  {
    return ptr_;
  }

  [[nodiscard]] HOST_DEVICE constexpr auto size() const noexcept -> size_type
  {
    return size_;
  }

  [[nodiscard]] HOST_DEVICE constexpr auto
  operator[](size_type index) const noexcept -> reference
  {
    assert(index < size_);
    return ptr_[index];
  }
};

template <class Itr, class SizeType>
Span(Itr, SizeType)
    -> Span<std::remove_reference_t<decltype(*std::declval<Itr&>())>>;
template <class T, std::size_t N> Span(T (&)[N]) -> Span<T>;

#endif // CUDA_PATH_TRACER_SPAN_HPP
