#ifndef CUDA_PATH_TRACER_SCENE_BUILDER_HPP
#define CUDA_PATH_TRACER_SCENE_BUILDER_HPP

#include "mesh.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

#include "scene.hpp"

#include <variant>
#include <vector>

class SceneBuilder {
  using CPUObject = std::variant<Sphere, Triangle, Mesh>;
  std::vector<CPUObject> objects_;
  std::vector<std::uint32_t> objects_material_indices_;

public:
  void add_object(CPUObject object, std::uint32_t material_index)
  {
    objects_.push_back(std::move(object));
    objects_material_indices_.push_back(material_index);
  }

  [[nodiscard]] auto build() const -> Aggregate;
};

#endif // CUDA_PATH_TRACER_SCENE_BUILDER_HPP
