#include "prelude.hpp"

#include <fmt/format.h>

[[noreturn]] void panic(std::string_view msg) noexcept
{
  fmt::print(stderr, "Panic: {}\n", msg);
  std::fflush(stderr);
  std::exit(1);
}