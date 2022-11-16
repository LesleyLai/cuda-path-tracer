#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "configurations.hpp"
#include "cuda_utils/cuda_buffer.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "resolution.hpp"
#include "scene.hpp"
#include "scene_description.hpp"

#include "denoising/edge_avoiding_a_trous_denoiser.hpp"

class FirstPersonCameraController;

enum class DisplayBufferType { final, color, normal, depth };

// SOA structure for each Ray path
struct Paths {
  cuda::Buffer<Ray> rays;
  cuda::Buffer<int> pixel_indices;

  // Changes the resolution of the frame buffer
  void resize_image(UResolution resolution);
};

class PathTracer {
public:
  int max_iterations = 1;

  EdgeAvoidingATrousDenoiser atrous_denoiser{};

private:
  Scene dev_scene_;

  Paths paths_;

  cuda::Buffer<glm::vec3> dev_color_buffer_;
  cuda::Buffer<glm::vec3> dev_normal_buffer_;
  cuda::Buffer<float> dev_depth_buffer_;

  cuda::Buffer<glm::vec3> dev_denoised_buffer_;
  cuda::Buffer<glm::vec3> dev_denoised_buffer2_;
  glm::vec3* path_trace_result_buffer_ = nullptr;

  GPUMesh bunny_;

  int iteration_ = 0;

public:
  PathTracer();

  void restart();

  [[nodiscard]] auto iteration() const noexcept -> int { return iteration_; }

  void resize_image(UResolution resolution);
  void create_buffers(UResolution resolution, const SceneDescription& scene);
  void path_trace(const Camera& camera, UResolution resolution);
  void denoise(UResolution resolution);

  void send_to_preview(uchar4* dev_pbo, UResolution resolution,
                       DisplayBufferType type = DisplayBufferType::final) const;
};
