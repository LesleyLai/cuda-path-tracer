#include "json.hpp"

#include <fmt/format.h>

namespace nlohmann {

template <> struct adl_serializer<glm::vec3> {
  static void from_json(const json& j, glm::vec3& v)
  {
    if (!j.is_array() || j.size() != 3) {
      throw std::runtime_error{"vec3 need to be 3d"};
    }
    v = glm::vec3{j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
  }
};

} // namespace nlohmann

namespace {

void read_materials(const nlohmann::json& json, SceneDescription& scene)
{
  const auto materials = json["materials"];
  if (!materials.is_array()) {
    throw std::runtime_error{"materials is not array"};
  }
  for (const auto& material : materials) {
    const auto name = material["name"].get<std::string>();
    const auto type = material["type"].get<std::string>();
    if (type == "lambertian") {
      const auto albedo = material["albedo"].get<glm::vec3>();
      scene.add_material(name, DiffuseMateral{albedo});
    } else if (type == "dielectric") {
      const auto refraction_index = material["refraction_index"].get<float>();
      scene.add_material(name, DielectricMaterial{refraction_index});
    } else if (type == "metal") {
      const auto albedo = material["albedo"].get<glm::vec3>();
      const auto fuzz = material["fuzz"].get<float>();
      scene.add_material(name, MetalMaterial{albedo, fuzz});
    } else {
      throw std::runtime_error{
          fmt::format("Unsupported material type {}", type)};
    }
  }
}

void read_surfaces(const nlohmann::json& json, SceneDescription& scene)
{
  const auto surfaces = json["surfaces"];
  if (!surfaces.is_array()) {
    throw std::runtime_error{"surfaces is not array"};
  }

  for (const auto& surface : surfaces) {
    const auto type = surface["type"].get<std::string>();
    const auto material = surface["material"].get<std::string>();

    if (type == "sphere") {
      const auto radius = surface["radius"].get<float>();
      const auto transform = surface["mat4"];
      const auto translate = transform["translate"];
      const auto pos = translate.get<glm::vec3>();

      scene.add_object(Sphere{pos, radius}, material);

    } else {
      throw std::runtime_error{
          fmt::format("Not supported surface type {}", type)};
    }
  }
}

} // anonymous namespace

[[nodiscard]] auto scene_from_json(const nlohmann::json& json)
    -> SceneDescription
{
  SceneDescription scene;
  read_materials(json, scene);
  read_surfaces(json, scene);

  return scene;
}
