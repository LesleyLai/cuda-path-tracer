#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "cuda_buffer.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "scene.hpp"
#include "scene_builder.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Camera;

enum DisplayBuffer { path_tracing, color, normal, position };

struct ATrousParameters {
  int filter_size = 10;
  float color_weight = 0.45f;
  float normal_weight = 0.30f;
  float position_weight = 0.25f;
};

class PathTracer {
public:
  int max_iterations = 10000;

  bool enable_denoising = true;
  ATrousParameters atrous_paramteters{};

private:
  Aggregate aggregate_;
  cuda::Buffer<Material> dev_mat_;
  cuda::Buffer<DiffuseMateral> dev_diffuse_mat_;
  cuda::Buffer<MetalMaterial> dev_metal_mat_;
  cuda::Buffer<DielectricMaterial> dev_dielectric_mat_;

  cuda::Buffer<glm::vec3> dev_color_buffer_;
  cuda::Buffer<glm::vec3> dev_normal_buffer_;
  cuda::Buffer<glm::vec3> dev_position_buffer_;

  cuda::Buffer<glm::vec3> dev_denoised_buffer_;
  cuda::Buffer<glm::vec3> dev_denoised_buffer2_;

  GPUMesh cube_;

  int iteration_ = 0;

  DisplayBuffer display_buffer_ = DisplayBuffer::path_tracing;

public:
  PathTracer();

  void restart();

  [[nodiscard]] auto iteration() const noexcept -> int { return iteration_; }

  void resize_image(unsigned int width, unsigned int height);

  void create_buffers(unsigned int width, unsigned int height,
                      const SceneBuilder& scene);
  void path_trace(uchar4* dev_pbo, const Camera& camera, unsigned int width,
                  unsigned int height);

  void set_display_type(DisplayBuffer display_type)
  {
    display_buffer_ = display_type;
  }
};
