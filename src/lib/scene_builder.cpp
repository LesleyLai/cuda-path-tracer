#include "scene_builder.hpp"

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

auto SceneBuilder::build() const -> Aggregate
{
  std::vector<GPUObject> gpu_objects;
  std::vector<Sphere> spheres;
  std::vector<Triangle> triangles;

  for (auto& object : objects_) {
    std::visit(overloaded{[&](Sphere sphere) {
                            gpu_objects.emplace_back(
                                ObjectType::sphere,
                                static_cast<std::uint32_t>(spheres.size()));
                            spheres.push_back(sphere);
                          },
                          [&](Triangle triangle) {
                            gpu_objects.emplace_back(
                                ObjectType::triangle,
                                static_cast<std::uint32_t>(triangles.size()));
                            triangles.push_back(triangle);
                          },
                          [&](const Mesh& /*mesh*/) {
                            gpu_objects.emplace_back(ObjectType::mesh, 0);
                          }},
               object);
  }

  Aggregate aggregate;
  auto gpu_objects_span =
      Span<const GPUObject>{gpu_objects.data(), gpu_objects.size()};
  aggregate.objects = cuda::create_buffer_from_cpu_data(gpu_objects_span);
  aggregate.object_count = std::size(gpu_objects);

  auto object_material_indices_span = Span<const std::uint32_t>{
      objects_material_indices_.data(), objects_material_indices_.size()};
  aggregate.object_material_indices =
      cuda::create_buffer_from_cpu_data(object_material_indices_span);

  auto sphere_span = Span<const Sphere>{spheres.data(), spheres.size()};
  aggregate.spheres = cuda::create_buffer_from_cpu_data(sphere_span);
  aggregate.sphere_count = std::size(spheres);

  auto triangle_span = Span<const Triangle>{triangles.data(), triangles.size()};
  aggregate.triangles = cuda::create_buffer_from_cpu_data(triangle_span);
  aggregate.triangle_count = std::size(triangles);
  return aggregate;
}
