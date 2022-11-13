#ifndef CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP
#define CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP

#include "camera.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

#include "scene.hpp"
#include "transform.hpp"

#include <map>
#include <variant>
#include <vector>

using Shape = std::variant<Sphere, Mesh>;
struct Object {
  Shape shape;
  Transform transform;
};

class SceneDescription {
  std::vector<Object> objects_;
  std::map<std::string, Material, std::less<>> material_map_;
  std::vector<std::string> objects_material_mapping_;

public:
  std::string filename;
  Camera camera;
  Resolution resolution;
  int spp;

  void add_material(const std::string& name, Material material)
  {
    material_map_.try_emplace(name, material);
  }

  void add_object(Shape shape, Transform transform,
                  const std::string& material_name)
  {
    if (const auto itr = material_map_.find(material_name);
        itr == material_map_.end()) {
      throw std::runtime_error{
          fmt::format("Cannot find material {}", material_name)};
    } else {
      objects_.push_back(Object{shape, transform});
      objects_material_mapping_.push_back(material_name);
    }
  }

  [[nodiscard]] auto build_scene() const -> Scene;
};

#endif // CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP
