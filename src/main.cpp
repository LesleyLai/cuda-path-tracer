#include "app.hpp"

#include <cstdio>

auto main() -> int
try {
  App app;
  app.main_loop();
} catch (const std::exception& e) {
  std::fputs(e.what(), stderr);
}
