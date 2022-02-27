#ifndef CUDA_PATH_TRACER_GLM_TEST_HELPER_HPP
#define CUDA_PATH_TRACER_GLM_TEST_HELPER_HPP

#include <catch2/catch.hpp>

#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/string_cast.hpp>

#include <glm/ext/matrix_relational.hpp>

#include <fmt/format.h>

namespace Catch {
template <glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
struct StringMaker<glm::mat<C, R, T, Q>> {
  static std::string convert(const glm::mat<C, R, T, Q>& mat)
  {
    return glm::to_string(mat);
  }
};

template <glm::length_t L, typename T, glm::qualifier Q>
struct StringMaker<glm::vec<L, T, Q>> {
  static std::string convert(const glm::vec<L, T, Q>& vec)
  {
    return glm::to_string(vec);
  }
};
} // namespace Catch

template <typename T> class GlmApproxMatcher : public Catch::MatcherBase<T> {
  T val_;
  typename T::value_type epsilon_;

public:
  explicit GlmApproxMatcher(T val, typename T::value_type epsilon)
      : val_{val}, epsilon_{epsilon}
  {
  }

  [[nodiscard]] auto match(const T& in) const -> bool override
  {
    return glm::all(glm::equal(val_, in, epsilon_));
  }

  [[nodiscard]] auto describe() const -> std::string override
  {
    return fmt::format("is approximately equal to {}", glm::to_string(val_));
  }
};

template <typename T>
[[nodiscard]] auto
ApproxEqual(T val, typename T::value_type epsilon =
                       std::numeric_limits<typename T::value_type>::epsilon() *
                       100) -> GlmApproxMatcher<T>
{
  return GlmApproxMatcher<T>{val, epsilon};
}

#endif // CUDA_PATH_TRACER_GLM_TEST_HELPER_HPP
