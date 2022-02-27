#ifndef CUDA_PATH_TRACER_MESH_HPP
#define CUDA_PATH_TRACER_MESH_HPP

#include <glm/glm.hpp>
#include <vector>

#include "cuda_utils/cuda_buffer.hpp"

struct Vertex {
  glm::vec3 position{};
};

struct Mesh {
  //  std::vector<Vertex> vertices;
  //  std::vector<std::uint32_t> indices;
};

struct GPUMesh {
  cuda::Buffer<Vertex> vertices;
  cuda::Buffer<std::uint32_t> indices;
  std::uint32_t indices_count = 0;
};

[[nodiscard]] auto load_obj(const char* filename) -> GPUMesh;

#endif // CUDA_PATH_TRACER_MESH_HPP
