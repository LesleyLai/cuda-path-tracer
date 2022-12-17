#include "json_parser.hpp"
#include "model_loader.hpp"

#include <fmt/format.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "lib/prelude.hpp"

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

template <> struct adl_serializer<glm::mat4> {
  static void from_json(const json& j, glm::mat4& m)
  {
    if (j.count("translate") == 1) {
      m = glm::translate(j["translate"].get<glm::vec3>());
    } else if (j.count("scale") == 1) {
      const auto scale = j["scale"];
      if (scale.is_number()) {
        m = glm::scale(glm::vec3{scale.get<float>()});
      } else {
        m = glm::scale(j["scale"].get<glm::vec3>());
      }
    } else {
      panic(fmt::format("Json parser: Unrecognized transform command: {}",
                        to_string(j)));
    }
  }
};

template <> struct adl_serializer<Transform> {
  static void from_json(const json& j, Transform& t)
  {
    glm::mat4 mat = glm::identity<glm::mat4>();
    if (j.is_object()) { // Single transform
      mat = j.get<glm::mat4>();
    } else if (j.is_array()) {
      // multiple transformation commands listed in order
      for (auto& elem : j) {
        mat = elem.get<glm::mat4>() * mat;
      }
    } else {
      panic("Transform must be either an object or an array!");
    }

    t = Transform{mat};
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

auto load_mesh_if_not_exist(SceneDescription& scene,
                            const std::string& filename) -> MeshRef
{
  if (auto maybe_mesh = scene.get_mesh(filename); maybe_mesh) {
    return maybe_mesh.value();
  }
  return scene.add_mesh(filename, load_obj(filename.c_str()));
}

void read_surfaces(const nlohmann::json& json,
                   const std::filesystem::path& file_dir,
                   SceneDescription& scene)
{
  const auto surfaces = json["surfaces"];
  if (!surfaces.is_array()) { panic("surfaces is not array"); }

  for (const auto& surface : surfaces) {
    const auto type = surface["type"].get<std::string>();
    const auto material = surface["material"].get<std::string>();

    if (type == "sphere") {
      const auto radius = surface["radius"].get<float>();
      const auto transform = surface["transform"].get<Transform>();
      scene.add_object(Sphere{glm::vec3{0}, radius}, transform, material);
    } else if (type == "mesh") {
      const auto transform = surface["transform"].get<Transform>();
      const auto filename = surface["filename"].get<std::string>();
      const std::filesystem::path file_path = canonical(file_dir / filename);
      const MeshRef mesh_ref =
          load_mesh_if_not_exist(scene, file_path.string());
      scene.add_object(mesh_ref, transform, material);
    } else {
      panic(fmt::format("Not supported surface type {}", type));
    }
  }
}

[[nodiscard]] auto json_from_file(const std::string& filename) -> nlohmann::json
{
  std::ifstream file{filename};
  if (not file.is_open()) {
    panic(fmt::format("Cannot open file {}\n", filename));
  }
  nlohmann::json json;
  file >> json;
  return json;
}

} // anonymous namespace

[[nodiscard]] auto scene_from_json(const std::string& filename)
    -> SceneDescription
{
  const nlohmann::json json = json_from_file(filename);

  const auto file_dir = std::filesystem::path{filename}.remove_filename();

  SceneDescription scene_desc;
  read_materials(json, scene_desc);
  read_surfaces(json, file_dir, scene_desc);

  const auto camera = json["camera"];
  if (!camera.is_object()) { panic("Camera is not an object!"); }

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