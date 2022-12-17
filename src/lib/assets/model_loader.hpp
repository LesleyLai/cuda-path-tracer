#ifndef CUDA_PATH_TRACER_MODEL_LOADER_HPP
#define CUDA_PATH_TRACER_MODEL_LOADER_HPP

#include "lib/mesh.hpp"

[[nodiscard]] auto load_obj(const char* filename) -> Mesh;

#endif // CUDA_PATH_TRACER_MODEL_LOADER_HPP
