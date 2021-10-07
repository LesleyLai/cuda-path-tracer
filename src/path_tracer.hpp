#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "cuda_buffer.hpp"
#include "ray.hpp"
#include "sphere.hpp"

struct Material {
  enum struct Type { Diffuse, Metal };
  Type type = Type::Diffuse;
  std::size_t index = 0;
};

struct DiffuseMateral {
  glm::vec3 albedo;
};

struct MetalMaterial {
  glm::vec3 albedo;
};

class PathTracer {
  cuda::Buffer<Sphere> dev_spheres_;
  cuda::Buffer<Material> dev_mat_;
  cuda::Buffer<DiffuseMateral> dev_diffuse_mat_;
  cuda::Buffer<MetalMaterial> dev_metal_mat_;

  cuda::Buffer<glm::vec3> dev_image_;

  std::size_t iteration_ = 0;

public:
  PathTracer();

  void reset();

  void resize_image(unsigned int width, unsigned int height);

  void create_buffers(unsigned int width, unsigned int height);
  void path_trace(uchar4* PBOpos, unsigned int width, unsigned int height);
};
