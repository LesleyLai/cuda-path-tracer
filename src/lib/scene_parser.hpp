#ifndef CUDA_PATH_TRACER_FILE_PARSER_HPP
#define CUDA_PATH_TRACER_FILE_PARSER_HPP

#include <string_view>

#include "configurations.hpp"
#include "scene_description.hpp"

[[nodiscard]] auto read_scene(std::string_view filename) -> SceneDescription;

#endif // CUDA_PATH_TRACER_FILE_PARSER_HPP
