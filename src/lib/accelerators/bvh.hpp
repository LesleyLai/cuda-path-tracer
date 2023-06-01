#ifndef CUDA_PATH_TRACER_BVH_HPP
#define CUDA_PATH_TRACER_BVH_HPP

#include "../aabb.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

struct Mesh;

// A zero `primitive_count` means that we have an inner node.
// - If it is a leaf, `first_child_or_primitive` is the index of the first
// triangle
// - If it is an inner node, `first_child_or_primitive` is the index of
// the left child
// And the index of the right child is `first_child_or_primitive + 1`
struct BVHNode {
  AABB aabb;
  std::uint32_t first_child_or_primitive = 0;
  std::uint32_t primitive_count = 0;

  [[nodiscard]] constexpr auto is_leaf() const noexcept
  {
    return primitive_count != 0;
  }
};

static_assert(sizeof(BVHNode) == 32);

auto bvh_from_mesh(const Mesh& mesh) -> std::vector<BVHNode>;

#endif // CUDA_PATH_TRACER_BVH_HPP
