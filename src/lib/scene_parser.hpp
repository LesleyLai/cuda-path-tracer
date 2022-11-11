#ifndef CUDA_PATH_TRACER_FILE_PARSER_HPP
#define CUDA_PATH_TRACER_FILE_PARSER_HPP

#include <filesystem>
#include <string_view>

#include "configurations.hpp"
#include "scene_description.hpp"

[[nodiscard]] auto read_scene(const CliConfigurations& configs,
                              const std::filesystem::path& asset_path)
    -> SceneDescription;

#endif // CUDA_PATH_TRACER_FILE_PARSER_HPP
