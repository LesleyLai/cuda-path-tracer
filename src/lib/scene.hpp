#ifndef CUDA_PATH_TRACER_SCENE_HPP
#define CUDA_PATH_TRACER_SCENE_HPP

#include "cuda_utils/cuda_buffer.hpp"

#include "aabb.hpp"
#include "mesh.hpp"
#include "span.hpp"
#include "sphere.hpp"
#include "transform.hpp"

enum class ObjectType : std::uint32_t { sphere, mesh };

struct GPUObject {
  ObjectType type{};
  std::uint32_t index{};

  Transform transform;
  AABB aabb;
};

// An aggregate contains object data on the GPU
struct Aggregate {
  std::size_t object_count = 0;
  cuda::Buffer<GPUObject> objects{};
  cuda::Buffer<std::uint32_t> object_material_indices{};

  std::size_t sphere_count = 0;
  cuda::Buffer<Sphere> spheres{};

  // Mesh
  cuda::Buffer<glm::vec3> positions;
  cuda::Buffer<std::uint32_t> indices;
  std::uint32_t indices_count = 0;
};

struct AggregateView {
  Span<const GPUObject> objects;
  const std::uint32_t* object_material_indices = nullptr;
  Span<const Sphere> spheres;

  const glm::vec3* positions;
  Span<const std::uint32_t> indices;

  AggregateView() = default;
  explicit AggregateView(const Aggregate& aggregate)
      : objects{aggregate.objects.data(), aggregate.object_count},
        object_material_indices{aggregate.object_material_indices.data()},
        spheres{aggregate.spheres.data(), aggregate.sphere_count},
        positions{aggregate.positions.data()}, indices{aggregate.indices.data(),
                                                       aggregate.indices_count}
  {
  }
};

struct Scene {
  Aggregate aggregate;
  cuda::Buffer<Material> materials;
};

#endif // CUDA_PATH_TRACER_SCENE_HPP
