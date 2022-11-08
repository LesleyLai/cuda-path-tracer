#ifndef CUDA_PATH_TRACER_FILE_PARSER_HPP
#define CUDA_PATH_TRACER_FILE_PARSER_HPP

#include <string_view>

#include "options.hpp"
#include "scene_description.hpp"

[[nodiscard]] auto read_scene(const Options& options) -> SceneDescription;

#endif // CUDA_PATH_TRACER_FILE_PARSER_HPP
