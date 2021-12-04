#ifndef CUDA_PATH_TRACER_GPU_SCENE_HPP
#define CUDA_PATH_TRACER_GPU_SCENE_HPP

#include "cuda_buffer.hpp"
#include "span.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

enum class ObjectType : std::uint32_t { sphere, triangle, mesh };
struct Object {
  ObjectType type{};
  std::uint32_t index{};
};

// An aggregate contains object data on the GPU
struct GPUAggregate {
  std::size_t object_count = 0;
  cuda::Buffer<Object> objects{};
  cuda::Buffer<std::uint32_t> object_material_indices{};

  std::size_t sphere_count = 0;
  cuda::Buffer<Sphere> spheres{};

  std::size_t triangle_count = 0;
  cuda::Buffer<Triangle> triangles{};
};

struct AggregateView {
  Span<const Object> objects;
  const std::uint32_t* object_material_indices = nullptr;
  Span<const Sphere> spheres;
  Span<const Triangle> triangles;

  AggregateView() = default;
  explicit AggregateView(const GPUAggregate& aggregate)
      : objects{aggregate.objects.data(), aggregate.object_count},
        object_material_indices{aggregate.object_material_indices.data()},
        spheres{aggregate.spheres.data(), aggregate.sphere_count},
        triangles{aggregate.triangles.data(), aggregate.triangle_count}
  {
  }
};

#endif // CUDA_PATH_TRACER_GPU_SCENE_HPP
