#include "prelude.hpp"

#include <spdlog/spdlog.h>

[[noreturn]] void panic(std::string_view msg) noexcept
{
  SPDLOG_CRITICAL("Panic: {}\n", msg);
  std::fflush(stderr);
  std::exit(1);
}