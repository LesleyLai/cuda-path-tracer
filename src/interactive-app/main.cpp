#include "app.hpp"

#include <cstdio>
#include <span>

auto main(int argc, char** argv) -> int
try {
  std::span<char*> args(argv, static_cast<std::size_t>(argc));
  App app{args};
  app.main_loop();
} catch (const std::exception& e) {
  std::fputs(e.what(), stderr);
}
