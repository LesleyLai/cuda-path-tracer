#ifndef CUDA_PATH_TRACER_JSON_PARSER_HPP
#define CUDA_PATH_TRACER_JSON_PARSER_HPP

#include "lib/scene_description.hpp"

#include <string>

[[nodiscard]] auto scene_from_json(const std::string& filename)
    -> SceneDescription;

#endif // CUDA_PATH_TRACER_JSON_PARSER_HPP
