#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "cuda_utils/cuda_buffer.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "scene.hpp"
#include "scene_description.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "denoising/edge_avoiding_a_trous_denoiser.hpp"

class Camera;

enum class DisplayBuffer { path_tracing, color, normal, depth };

class PathTracer {
public:
  int max_iterations = 1;

  bool enable_denoising = false;

  EdgeAvoidingATrousDenoiser atrous_denoiser{};

private:
  Scene dev_scene_;

  cuda::Buffer<glm::vec3> dev_color_buffer_;
  cuda::Buffer<glm::vec3> dev_normal_buffer_;
  cuda::Buffer<float> dev_depth_buffer_;

  cuda::Buffer<glm::vec3> dev_denoised_buffer_;
  cuda::Buffer<glm::vec3> dev_denoised_buffer2_;
  glm::vec3* path_trace_result_buffer_ = nullptr;

  GPUMesh bunny_;

  int iteration_ = 0;

  DisplayBuffer display_buffer_ = DisplayBuffer::path_tracing;

public:
  PathTracer();

  void restart();

  [[nodiscard]] auto iteration() const noexcept -> int { return iteration_; }

  void resize_image(unsigned int width, unsigned int height);

  void create_buffers(unsigned int width, unsigned int height,
                      const SceneDescription& scene);
  void path_trace(const Camera& camera, unsigned int width,
                  unsigned int height);
  void send_to_preview(uchar4* dev_pbo, unsigned int width,
                       unsigned int height) const;

  void set_display_type(DisplayBuffer display_type)
  {
    display_buffer_ = display_type;
  }
};
