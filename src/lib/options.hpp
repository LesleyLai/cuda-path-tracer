#ifndef CUDA_PATH_TRACER_OPTIONS_HPP
#define CUDA_PATH_TRACER_OPTIONS_HPP

#include <string>

struct Options {
  bool is_interactive = false;
  std::string filename = nullptr;
};

[[nodiscard]] auto parse_cli_args(int argc, char** argv) -> Options;

#endif // CUDA_PATH_TRACER_OPTIONS_HPP
