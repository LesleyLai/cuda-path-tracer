#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "cuda_buffer.hpp"
#include "gpu_scene.hpp"
#include "material.hpp"
#include "ray.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Camera;

struct Vertex {
  glm::vec3 position;
};

struct Mesh {
  cuda::Buffer<Vertex> vertices;
  cuda::Buffer<std::uint32_t> indices;
  std::uint32_t indices_count;
};

class PathTracer {
public:
  int max_iterations = 10000;

private:
  GPUAggregate aggregate_;
  cuda::Buffer<Material> dev_mat_;
  cuda::Buffer<DiffuseMateral> dev_diffuse_mat_;
  cuda::Buffer<MetalMaterial> dev_metal_mat_;
  cuda::Buffer<DielectricMaterial> dev_dielectric_mat_;

  cuda::Buffer<glm::vec3> dev_image_;

  Mesh cube_;

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
