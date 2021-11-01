#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "cuda_buffer.hpp"
#include "ray.hpp"
#include "sphere.cuh"

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

class Camera;

class PathTracer {
public:
  int max_iterations = 10000;

private:
  cuda::Buffer<Sphere> dev_spheres_;
  cuda::Buffer<Triangle> dev_triangles_;

  cuda::Buffer<Material> dev_mat_;
  cuda::Buffer<DiffuseMateral> dev_diffuse_mat_;
  cuda::Buffer<MetalMaterial> dev_metal_mat_;
  cuda::Buffer<DielectricMaterial> dev_dielectric_mat_;

  cuda::Buffer<glm::vec3> dev_image_;

  int iteration_ = 0;

public:
  PathTracer();

  void restart();

  [[nodiscard]] auto iteration() const noexcept -> int { return iteration_; }

  void resize_image(unsigned int width, unsigned int height);

  void create_buffers(unsigned int width, unsigned int height);
  void path_trace(uchar4* dev_pbo, const Camera& camera, unsigned int width,
                  unsigned int height);
};
