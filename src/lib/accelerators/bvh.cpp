#include "bvh.hpp"
#include "../mesh.hpp"

#include <lib/prelude.hpp>
#include <memory>
#include <span>

namespace {

struct CPUBVHNode {
  AABB aabb;

  CPUBVHNode() = default;
  virtual ~CPUBVHNode() = default;
  explicit CPUBVHNode(AABB aabb_) : aabb{aabb_} {}

  [[nodiscard]] virtual auto is_leaf() const -> bool = 0;
};

struct CPUBVHLeaf : CPUBVHNode {
  unsigned int triangle_index_begin = 0;

  CPUBVHLeaf(AABB aabb, unsigned int i)
      : CPUBVHNode{aabb}, triangle_index_begin{i}
  {
  }

  [[nodiscard]] auto is_leaf() const -> bool final { return true; }
};

struct CPUBVHInner : CPUBVHNode {
  std::shared_ptr<CPUBVHNode> left;
  std::shared_ptr<CPUBVHNode> right;

  CPUBVHInner(std::shared_ptr<CPUBVHNode> left_,
              std::shared_ptr<CPUBVHNode> right_)
      : left{std::move(left_)}, right{std::move(right_)}
  {
    aabb = aabb_union(left->aabb, right->aabb);
  }

  [[nodiscard]] auto is_leaf() const -> bool final { return false; }
};

[[nodiscard]] auto
pick_split_axis(std::span<std::shared_ptr<CPUBVHLeaf>> leaves) -> unsigned int
{
  AABB centroid_bound;
  for (const auto& leave : leaves) {
    centroid_bound = centroid_bound.enclose(leave->aabb.center());
  }
  return centroid_bound.max_extent();
}

[[nodiscard]] auto
cpu_bvh_from_leaves(std::span<std::shared_ptr<CPUBVHLeaf>> leaves)
    -> std::shared_ptr<CPUBVHNode>
{
  const auto axis = pick_split_axis(leaves);

  std::shared_ptr<CPUBVHNode> left;
  std::shared_ptr<CPUBVHNode> right;

  if (leaves.empty()) {
    panic("Shouldn't happen!");
  } else if (leaves.size() == 1) {
    return std::move(leaves.front());
  } else if (leaves.size() == 2) {
    left = std::move(leaves[0]);
    right = std::move(leaves[1]);
    if (left->aabb.center()[axis] > right->aabb.center()[axis]) {
      std::swap(left, right);
    }
  } else {
    // Split half-by-half
    const auto mid = static_cast<std::ptrdiff_t>(leaves.size() / 2);
    std::ranges::nth_element(leaves.begin(), leaves.begin() + mid, leaves.end(),
                             {}, [&](const std::shared_ptr<CPUBVHLeaf>& leaf) {
                               return leaf->aabb.center()[axis];
                             });

    std::span left_leaves(leaves.begin(), leaves.begin() + mid);
    std::span right_leaves(leaves.begin() + mid, leaves.end());

    left = cpu_bvh_from_leaves(left_leaves);
    right = cpu_bvh_from_leaves(right_leaves);
  }

  return std::make_shared<CPUBVHInner>(left, right);
}

[[nodiscard]] auto cpu_bvh_from_mesh(const Mesh& mesh)
    -> std::shared_ptr<CPUBVHNode>
{
  std::vector<std::shared_ptr<CPUBVHLeaf>> leaves;
  for (unsigned int i = 0; i < mesh.indices.size(); i += 3) {
    const std::uint32_t index0 = mesh.indices[i];
    const std::uint32_t index1 = mesh.indices[i + 1];
    const std::uint32_t index2 = mesh.indices[i + 2];
    const glm::vec3 p1 = mesh.positions[index0];
    const glm::vec3 p2 = mesh.positions[index1];
    const glm::vec3 p3 = mesh.positions[index2];

    const AABB aabb = AABB{}.enclose(p1).enclose(p2).enclose(p3);
    leaves.push_back(std::make_shared<CPUBVHLeaf>(aabb, i));
  }

  if (leaves.empty()) {
    panic("Cannot create BVH for empty mesh");
  } else if (leaves.size() == 1) {
    return std::move(leaves.front());
  } else {
    return cpu_bvh_from_leaves(leaves);
  }
}

void populate_linear_bvh(std::vector<BVHNode>& linear_bvh,
                         const CPUBVHNode& node)
{
  if (node.is_leaf()) {
    const auto leaf = static_cast<const CPUBVHLeaf&>(node);
    linear_bvh.push_back(BVHNode{
        .aabb = leaf.aabb,
        .is_leaf = true,
        .data = {.leaf = {.triangle_index_begin = leaf.triangle_index_begin}}});
  } else { // Depth first: always go to left first
    const auto inner = static_cast<const CPUBVHInner&>(node);

    const auto current_index = static_cast<std::uint32_t>(linear_bvh.size());
    linear_bvh.push_back(BVHNode{.aabb = inner.aabb, .is_leaf = false});
    const auto left_index = current_index + 1;
    populate_linear_bvh(linear_bvh, *inner.left);
    const auto right_index = static_cast<std::uint32_t>(linear_bvh.size());
    populate_linear_bvh(linear_bvh, *inner.right);

    linear_bvh[current_index].data.inner = {
        .left_index = left_index,
        .right_index = right_index,
    };
  }
}

} // anonymous namespace

auto bvh_from_mesh(const Mesh& mesh) -> std::vector<BVHNode>
{
  auto root_node = cpu_bvh_from_mesh(mesh);

  std::vector<BVHNode> linear_bvh;
  const auto size = mesh.triangle_count() * 2 - 1;
  linear_bvh.reserve(size);
  populate_linear_bvh(linear_bvh, *root_node);

  return linear_bvh;
}