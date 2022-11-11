#ifndef CUDA_PATH_TRACER_ASSETS_HPP
#define CUDA_PATH_TRACER_ASSETS_HPP

#include <filesystem>

auto locate_asset_path(const std::filesystem::path& current_path)
    -> std::filesystem::path;

#endif // CUDA_PATH_TRACER_ASSETS_HPP
