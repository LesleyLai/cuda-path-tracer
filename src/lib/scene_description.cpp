#include "scene_description.hpp"
#include "prelude.hpp"

#include <algorithm>

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

auto SceneDescription::build_scene() const -> Scene
{
  std::vector<GPUObject> gpu_objects;
  std::vector<Sphere> spheres;

  const auto bunny = load_obj("../../assets/models/bunny.obj");

  for (auto& object : objects_) {
    ObjectType object_type;
    std::uint32_t index = 0;
    AABB aabb;

    std::visit(
        overloaded{[&](Sphere sphere) {
                     object_type = ObjectType::sphere;
                     index = static_cast<std::uint32_t>(spheres.size());

                     const auto transformed_origin =
                         transform_point(object.transform, sphere.center);
                     const float transformed_radius =
                         glm::length(transform_vector(
                             object.transform, glm::vec3(1.0, 0.0, 0.0))) *
                         sphere.radius;
                     aabb = AABB{.min = transformed_origin -
                                        glm::vec3(transformed_radius),
                                 .max = transformed_origin +
                                        glm::vec3(transformed_radius)};

                     spheres.push_back(sphere);
                   },
                   [&](const Mesh& /*mesh*/) {
                     object_type = ObjectType::mesh;
                     index = 0;
                     aabb = transform_aabb(object.transform, bunny.aabb);
                   }},
        object.shape);

    gpu_objects.push_back(GPUObject{.type = object_type,
                                    .index = index,
                                    .transform = object.transform,
                                    .aabb = aabb});
  }

  Aggregate aggregate;
  auto gpu_objects_span =
      Span<const GPUObject>{gpu_objects.data(), gpu_objects.size()};
  aggregate.objects = cuda::make_buffer_from_cpu_data(gpu_objects_span);
  aggregate.object_count = std::size(gpu_objects);

  std::vector<Material> materials_vec;
  materials_vec.reserve(material_map_.size());
  std::map<std::string, std::uint32_t, std::less<>> material_indices_map;
  for (const auto& [name, material] : material_map_) {
    materials_vec.push_back(material);
    material_indices_map.insert(
        {name, static_cast<std::uint32_t>(materials_vec.size() - 1)});
  }

  const auto material_index_from_name =
      [&material_indices_map](
          const std::string& material_name) -> std::uint32_t {
    if (const auto itr = material_indices_map.find(material_name);
        itr != material_indices_map.end()) {
      return itr->second;
    } else {
      panic(fmt::format("Cannot find material {}", material_name));
    }
  };

  std::vector<std::uint32_t> objects_material_indices(
      objects_material_mapping_.size());
  std::ranges::transform(objects_material_mapping_,
                         objects_material_indices.begin(),
                         material_index_from_name);

  auto object_material_indices_span = Span<const std::uint32_t>{
      objects_material_indices.data(), objects_material_indices.size()};
  aggregate.object_material_indices =
      cuda::make_buffer_from_cpu_data(object_material_indices_span);

  auto sphere_span = Span<const Sphere>{spheres.data(), spheres.size()};
  aggregate.spheres = cuda::make_buffer_from_cpu_data(sphere_span);
  aggregate.sphere_count = std::size(spheres);

  // Mesh stuff
  auto bunny_indices =
      Span<const std::uint32_t>{bunny.indices.data(), bunny.indices.size()};

  aggregate.positions = cuda::make_buffer_from_cpu_data(
      Span{bunny.positions.data(), bunny.positions.size()});
  aggregate.indices = cuda::make_buffer_from_cpu_data(bunny_indices);
  aggregate.indices_count = static_cast<std::uint32_t>(bunny_indices.size());

  cuda::Buffer<Material> materials = cuda::make_buffer_from_cpu_data(
      Span(materials_vec.data(), materials_vec.size()));

  return Scene{std::move(aggregate), std::move(materials)};
}
