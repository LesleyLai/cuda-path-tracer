#ifndef CUDA_PATH_TRACER_CONFIGURATIONS_HPP
#define CUDA_PATH_TRACER_CONFIGURATIONS_HPP

#include <optional>
#include <string>

/**
 * @brief Configurations get from command line
 */
struct CliConfigurations {
  std::string filename;
  std::optional<int> spp = std::nullopt;
  std::optional<std::string> output_filename;
};

[[nodiscard]] auto parse_cli_args(int argc, char** argv) -> CliConfigurations;

#endif // CUDA_PATH_TRACER_CONFIGURATIONS_HPP
