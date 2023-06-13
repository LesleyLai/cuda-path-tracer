#include "bvh.hpp"
#include "../mesh.hpp"

#include <lib/prelude.hpp>
#include <memory>
#include <queue>
#include <span>

#include <fmt/format.h>

namespace {

struct CPUBVHNode {
  AABB aabb;

  CPUBVHNode() = default;
  virtual ~CPUBVHNode() = default;
  explicit CPUBVHNode(AABB aabb_) : aabb{aabb_} {}

  [[nodiscard]] virtual auto is_leaf() const -> bool = 0;

  [[nodiscard]] virtual auto to_linear_node() const -> BVHNode = 0;
};

struct CPUBVHLeaf : CPUBVHNode {
  unsigned int triangle_index_begin = 0;

  CPUBVHLeaf(AABB aabb_, unsigned int i)
      : CPUBVHNode{aabb_}, triangle_index_begin{i}
  {
  }

  [[nodiscard]] auto is_leaf() const -> bool final { return true; }

  [[nodiscard]] auto to_linear_node() const -> BVHNode final
  {
    return BVHNode{.aabb = aabb,
                   .first_child_or_primitive = triangle_index_begin,
                   .primitive_count = 1};
  }
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

  [[nodiscard]] auto to_linear_node() const -> BVHNode final
  {
    return BVHNode{.aabb = aabb, .primitive_count = 0};
  }
};

[[nodiscard]] auto
calculate_centroid_bound(std::span<std::shared_ptr<CPUBVHLeaf>> leaves) -> AABB
{
  AABB bound;
  for (const auto& leave : leaves) {
    bound = bound.enclose(leave->aabb.center());
  }
  return bound;
}

auto split_by_SAH(const std::span<std::shared_ptr<CPUBVHLeaf>>& leaves,
                  const AABB& centroid_bound, const int axis)
    -> std::shared_ptr<CPUBVHNode>;

[[nodiscard]] auto
cpu_bvh_from_leaves(std::span<std::shared_ptr<CPUBVHLeaf>> leaves)
    -> std::shared_ptr<CPUBVHNode>
{
  const AABB centroid_bound = calculate_centroid_bound(leaves);
  const auto axis = centroid_bound.max_extent();

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
    return std::make_shared<CPUBVHInner>(left, right);
  } else {
    decltype(leaves)::iterator partition_point;

    if (leaves.size() <= 4) {
      //  Split half-by-half
      partition_point =
          leaves.begin() + static_cast<std::ptrdiff_t>(leaves.size() / 2);
      std::ranges::nth_element(
          leaves.begin(), partition_point, leaves.end(), {},
          [&](const auto& leaf) { return leaf->aabb.center()[axis]; });

      left = cpu_bvh_from_leaves({leaves.begin(), partition_point});
      right = cpu_bvh_from_leaves({partition_point, leaves.end()});
      return std::make_shared<CPUBVHInner>(left, right);
    } else {
      return split_by_SAH(leaves, centroid_bound, axis);
    }
  }
}

auto split_by_SAH(const std::span<std::shared_ptr<CPUBVHLeaf>>& leaves,
                  const AABB& centroid_bound, const int axis)
    -> std::shared_ptr<CPUBVHNode>
{
  // Create BucketInfo for SAH partition buckets
  constexpr size_t buckets_count = 12;
  struct BucketInfo {
    int count = 0;
    AABB bounds;
  };
  BucketInfo buckets[buckets_count];
  AABB bound;

  auto find_bucket_index = [&](const CPUBVHLeaf& leave) {
    size_t b =
        static_cast<int>(static_cast<float>(buckets_count) *
                         centroid_bound.offset(leave.aabb.center())[axis]);
    if (b == buckets_count) b = buckets_count - 1;
    return b;
  };

  for (const auto& leave : leaves) {
    const auto b = find_bucket_index(*leave);
    buckets[b].count++;
    buckets[b].bounds = aabb_union(buckets[b].bounds, leave->aabb);
    bound = aabb_union(bound, leave->aabb);
  }

  // Compute costs for splitting after each bucket
  float cost[buckets_count - 1];
  for (size_t i = 0; i < buckets_count - 1; ++i) {
    AABB b0, b1;
    int count0 = 0, count1 = 0;
    for (size_t j = 0; j <= i; ++j) {
      b0 = aabb_union(b0, buckets[j].bounds);
      count0 += buckets[j].count;
    }
    for (size_t j = i + 1; j < buckets_count; ++j) {
      b1 = aabb_union(b1, buckets[j].bounds);
      count1 += buckets[j].count;
    }
    cost[i] = .125f + (static_cast<float>(count0) * b0.surface_area() +
                       static_cast<float>(count1) * b1.surface_area()) /
                          bound.surface_area();
  }

  // Find bucket to split at that minimizes SAH metric
  float min_cost = cost[0];
  size_t min_cost_split_bucket = 0;
  for (size_t i = 1; i < buckets_count - 1; ++i) {
    if (cost[i] < min_cost) {
      min_cost = cost[i];
      min_cost_split_bucket = i;
    }
  }

  // Split primitives at selected SAH bucket

  const auto partition_point =
      std::ranges::partition(leaves, [=](const std::shared_ptr<CPUBVHLeaf>&
                                             leave) {
        return find_bucket_index(*leave) <= min_cost_split_bucket;
      }).begin();

  return std::make_shared<CPUBVHInner>(
      cpu_bvh_from_leaves({leaves.begin(), partition_point}),
      cpu_bvh_from_leaves({partition_point, leaves.end()}));
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

} // anonymous namespace

auto bvh_from_mesh(const Mesh& mesh) -> std::vector<BVHNode>
{
  auto root_node = cpu_bvh_from_mesh(mesh);

  std::vector<BVHNode> linear_bvh;
  const auto size = mesh.triangle_count() * 2 - 1;
  linear_bvh.reserve(size);

  // Breath-first flatten the tree
  struct CPUNodeToProcess {
    const CPUBVHNode* node = nullptr;
    std::uint32_t linear_index = 0;
  };
  std::queue<CPUNodeToProcess> nodes_to_process;

  auto push = [&](const CPUBVHNode& node) {
    nodes_to_process.push(CPUNodeToProcess{
        .node = &node,
        .linear_index = static_cast<std::uint32_t>(linear_bvh.size())});
    linear_bvh.push_back(node.to_linear_node());
  };

  push(*root_node);

  while (!nodes_to_process.empty()) {
    const auto [current_node, index] = nodes_to_process.front();

    if (!current_node->is_leaf()) {
      const auto* inner_node = static_cast<const CPUBVHInner*>(current_node);
      const auto left_index = static_cast<std::uint32_t>(linear_bvh.size());

      // update children indices
      linear_bvh[index].first_child_or_primitive = left_index;

      push(*inner_node->left);
      push(*inner_node->right);
    }

    nodes_to_process.pop();
  }

  return linear_bvh;
}