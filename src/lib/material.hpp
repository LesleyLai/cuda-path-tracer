#ifndef CUDA_PATH_TRACER_MATERIAL_HPP
#define CUDA_PATH_TRACER_MATERIAL_HPP

struct Material {
  enum struct Type { Diffuse, Metal, Dielectric };
  Type type = Type::Diffuse;
  std::size_t index = 0;
};

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

#endif // CUDA_PATH_TRACER_MATERIAL_HPP
