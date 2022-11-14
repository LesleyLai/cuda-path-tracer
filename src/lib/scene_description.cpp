#include "scene_description.hpp"

#include <algorithm>

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

auto SceneDescription::build_scene() const -> Scene
{
  std::vector<GPUObject> gpu_objects;
  std::vector<Sphere> spheres;

  for (auto& object : objects_) {
    ObjectType object_type;
    std::uint32_t index = 0;

    std::visit(overloaded{[&](Sphere sphere) {
                            object_type = ObjectType::sphere;
                            index = static_cast<std::uint32_t>(spheres.size());

                            spheres.push_back(sphere);
                          },
                          [&](const Mesh& /*mesh*/) {
                            object_type = ObjectType::mesh;
                            index = 0;
                          }},
               object.shape);

    gpu_objects.emplace_back(object_type, index, object.transform);
  }

  Aggregate aggregate;
  auto gpu_objects_span =
      Span<const GPUObject>{gpu_objects.data(), gpu_objects.size()};
  aggregate.objects = cuda::create_buffer_from_cpu_data(gpu_objects_span);
  aggregate.object_count = std::size(gpu_objects);

  std::vector<Material> materials_vec;
  materials_vec.reserve(material_map_.size());
  std::map<std::string, std::uint32_t, std::less<>> material_indices_map;
  for (const auto& [name, material] : material_map_) {
    materials_vec.push_back(material);
    material_indices_map.insert(
        {name, static_cast<std::uint32_t>(materials_vec.size() - 1)});
  }

  std::vector<std::uint32_t> objects_material_indices(
      objects_material_mapping_.size());
  std::ranges::transform(
      objects_material_mapping_, objects_material_indices.begin(),
      [&](const std::string& material_name) -> std::uint32_t {
        if (const auto itr = material_indices_map.find(material_name);
            itr != material_indices_map.end()) {
          return itr->second;
        } else {
          throw std::runtime_error{
              fmt::format("Cannot find material {}", material_name)};
        }
      });

  auto object_material_indices_span = Span<const std::uint32_t>{
      objects_material_indices.data(), objects_material_indices.size()};
  aggregate.object_material_indices =
      cuda::create_buffer_from_cpu_data(object_material_indices_span);

  auto sphere_span = Span<const Sphere>{spheres.data(), spheres.size()};
  aggregate.spheres = cuda::create_buffer_from_cpu_data(sphere_span);
  aggregate.sphere_count = std::size(spheres);

  cuda::Buffer<Material> materials = cuda::create_buffer_from_cpu_data(
      Span(materials_vec.data(), materials_vec.size()));

  return Scene{std::move(aggregate), std::move(materials)};
}
