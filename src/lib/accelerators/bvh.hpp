#ifndef CUDA_PATH_TRACER_BVH_HPP
#define CUDA_PATH_TRACER_BVH_HPP

#include "../aabb.hpp"
#include <cstdint>
#include <vector>

struct Mesh;

struct BVHNode {
  AABB aabb;
  bool is_leaf = false;

  union {
    struct {
      std::uint32_t triangle_index_begin =
          UINT32_MAX; // Index in the index buffer for the index of the first
                      // vertex in triangle
    } leaf;
    struct {
      std::uint32_t left_index = UINT32_MAX;  // Index of the left node
      std::uint32_t right_index = UINT32_MAX; // Index of the right node
    } inner;
  } data;
};

auto bvh_from_mesh(const Mesh& mesh) -> std::vector<BVHNode>;

#endif // CUDA_PATH_TRACER_BVH_HPP
