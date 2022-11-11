#ifndef CUDA_PATH_TRACER_CONFIGURATIONS_HPP
#define CUDA_PATH_TRACER_CONFIGURATIONS_HPP

#include <optional>
#include <string>

struct CliConfigurations {
  bool is_interactive = false;
  std::string filename;
  std::optional<int> spp = std::nullopt;
};

[[nodiscard]] auto parse_cli_args(int argc, char** argv) -> CliConfigurations;

#endif // CUDA_PATH_TRACER_CONFIGURATIONS_HPP
