#ifndef CUDA_PATH_TRACER_JSON_HPP
#define CUDA_PATH_TRACER_JSON_HPP

#include "scene_description.hpp"

#include <nlohmann/json.hpp>

[[nodiscard]] auto scene_from_json(const nlohmann::json& json)
    -> SceneDescription;

#endif // CUDA_PATH_TRACER_JSON_HPP
