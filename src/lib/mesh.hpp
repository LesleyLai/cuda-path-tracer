#ifndef CUDA_PATH_TRACER_MESH_HPP
#define CUDA_PATH_TRACER_MESH_HPP

#include <glm/glm.hpp>
#include <vector>

#include "aabb.hpp"

struct Mesh {
  std::vector<glm::vec3> positions;
  std::vector<std::uint32_t> indices;
  AABB aabb;
};

#endif // CUDA_PATH_TRACER_MESH_HPP
