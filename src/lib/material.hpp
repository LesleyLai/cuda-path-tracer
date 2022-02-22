#ifndef CUDA_PATH_TRACER_MATERIAL_HPP
#define CUDA_PATH_TRACER_MATERIAL_HPP

#include <glm/vec3.hpp>

struct DiffuseMateral {
  glm::vec3 albedo;
};

struct MetalMaterial {
  glm::vec3 albedo;
  float fuzz = 0;
};

struct DielectricMaterial {
  float refraction_index = 1.0;
};

struct Material {
  enum struct Type { Diffuse, Metal, Dielectric };
  Type type = Type::Diffuse;
  union Data {
    DiffuseMateral diffuse;
    MetalMaterial metal;
    DielectricMaterial dielectric;

    Data(DiffuseMateral diffuse_) : diffuse{diffuse_} {}
    Data(MetalMaterial metal_) : metal{metal_} {}
    Data(DielectricMaterial dielectric_) : dielectric{dielectric_} {}
  } data;

  Material(DiffuseMateral diffuse) : type{Type::Diffuse}, data{diffuse} {}
  Material(MetalMaterial metal) : type{Type::Metal}, data{metal} {}
  Material(DielectricMaterial dielectric)
      : type{Type::Dielectric}, data{dielectric}
  {
  }
};

static_assert(std::is_trivially_destructible_v<Material>);

#endif // CUDA_PATH_TRACER_MATERIAL_HPP
