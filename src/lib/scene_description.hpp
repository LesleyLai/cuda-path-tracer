#ifndef CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP
#define CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP

#include "camera.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "sphere.hpp"

#include "scene.hpp"
#include "transform.hpp"

#include "prelude.hpp"

#include <map>
#include <optional>
#include <variant>
#include <vector>

using MeshRef = std::reference_wrapper<const Mesh>;

using Shape = std::variant<Sphere, MeshRef>;
struct Object {
  Shape shape;
  Transform transform;
};

class SceneDescription {
  std::vector<Object> objects_;
  std::map<std::string, Material, std::less<>> material_map_;
  std::vector<std::string> objects_material_mapping_;
  std::map<std::string, Mesh, std::less<>> mesh_map_;

public:
  std::string filename;
  Camera camera;
  Resolution resolution;
  int spp;

  void add_material(const std::string& name, Material material);

  void add_object(Shape shape, Transform transform,
                  const std::string& material_name);

  auto get_mesh(const std::string& name) -> std::optional<MeshRef>;

  auto add_mesh(std::string name, Mesh&& mesh) -> MeshRef;

  [[nodiscard]] auto build_scene() const -> Scene;
};

#endif // CUDA_PATH_TRACER_SCENE_DESCRIPTION_HPP
