#ifndef CUDA_PATH_TRACER_CLI_HPP
#define CUDA_PATH_TRACER_CLI_HPP

#include "lib/configurations.hpp"

#include <filesystem>

void execute_cli_version(const CliConfigurations& configs,
                         std::filesystem::path asset_path);

#endif // CUDA_PATH_TRACER_CLI_HPP
