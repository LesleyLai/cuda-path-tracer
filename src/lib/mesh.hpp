#ifndef CUDA_PATH_TRACER_MESH_HPP
#define CUDA_PATH_TRACER_MESH_HPP

#include <glm/glm.hpp>
#include <vector>

#include "cuda_utils/cuda_buffer.hpp"

struct Vertex {
  glm::vec3 position{};
};

struct Mesh {
  // Vertices
  std::vector<glm::vec3> positions;
  // Indices
  std::vector<std::uint32_t> indices;
};

[[nodiscard]] auto load_obj(const char* filename) -> Mesh;

#endif // CUDA_PATH_TRACER_MESH_HPP
