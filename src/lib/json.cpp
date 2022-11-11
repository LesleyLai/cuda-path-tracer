#include "json.hpp"

#include <fmt/format.h>

#include <glm/gtc/matrix_transform.hpp>

#include "prelude.hpp"

namespace nlohmann {

template <> struct adl_serializer<glm::vec3> {
  static void from_json(const json& j, glm::vec3& v)
  {
    if (!j.is_array() || j.size() != 3) { panic("vec3 need to be 3d"); }
    v = glm::vec3{j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
  }
};

template <> struct adl_serializer<Resolution> {
  static void from_json(const json& j, Resolution& res)
  {
    if (!j.is_array() || j.size() != 2) { panic("resolution need to be 2d"); }
    res = Resolution{j[0].get<int>(), j[1].get<int>()};
  }
};

} // namespace nlohmann

namespace {

void read_materials(const nlohmann::json& json, SceneDescription& scene)
{
  const auto materials = json["materials"];
  if (!materials.is_array()) { panic("materials is not array!"); }
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
      panic(fmt::format("Unsupported material type {}", type));
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
      const auto transform = surface["transform"];
      const auto translate = transform["translate"].get<glm::vec3>();

      scene.add_object(Sphere{glm::vec3{0}, radius},
                       Transform{glm::translate(glm::mat4(1), translate)},
                       material);

    } else {
      throw std::runtime_error{
          fmt::format("Not supported surface type {}", type)};
    }
  }
}

} // anonymous namespace

[[nodiscard]] auto scene_from_json([[maybe_unused]] const Options& options,
                                   const nlohmann::json& json)
    -> SceneDescription
{
  SceneDescription scene_desc;
  read_materials(json, scene_desc);
  read_surfaces(json, scene_desc);

  const auto camera = json["camera"];
  if (!camera.is_object()) {
    throw std::runtime_error{"camera is not an object!"};
  }

  if (const auto itr = camera.find("resolution"); itr != camera.end()) {
    scene_desc.resolution = itr->get<Resolution>();
  }

  std::optional<int> spp = std::nullopt;

  if (const auto sampler_itr = json.find("sampler");
      sampler_itr != json.end()) {
    const auto& sampler = *sampler_itr;
    if (!sampler.is_object()) { panic("Sampler is not an object!"); }
    spp = sampler["samples"].get<int>();
  }
  scene_desc.spp = spp.value_or(1);

  return scene_desc;
}
