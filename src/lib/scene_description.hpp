#ifndef CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP
#define CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP

#include "material.hpp"
#include "mesh.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

#include "scene.hpp"

#include <map>
#include <variant>
#include <vector>

class SceneDescription {
  using CPUObject = std::variant<Sphere, Triangle, Mesh>;
  std::vector<CPUObject> objects_;
  std::map<std::string, Material, std::less<>> material_map_;
  std::vector<std::string> objects_material_mapping_;

public:
  void add_material(const std::string& name, Material material)
  {
    material_map_.insert({name, material});
  }

  void add_object(CPUObject object, const std::string& material_name)
  {
    if (const auto itr = material_map_.find(material_name);
        itr == material_map_.end()) {
      throw std::runtime_error{
          fmt::format("Cannot find material {}", material_name)};
    } else {
      objects_.push_back(object);
      objects_material_mapping_.push_back(material_name);
    }
  }

  [[nodiscard]] auto build() const -> Scene;
};

#endif // CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP
