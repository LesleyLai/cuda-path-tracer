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

using Normal = glm::vec3;

// SOA structure for each Ray path
struct Paths {
  cuda::Buffer<Ray> rays;
  cuda::Buffer<int> pixel_indices;
  cuda::Buffer<glm::vec3> color_buffer;
  cuda::Buffer<Normal> normal_buffer;
  cuda::Buffer<float> depth_buffer;
  cuda::Buffer<std::uint8_t> bounces_left_buffer;

  // Changes the resolution of the frame buffer
  void resize_image(UResolution resolution);
};

struct PathsView {
  unsigned int paths_count = 0;
  Ray* rays = nullptr;
  int* pixel_indices = nullptr;
  glm::vec3* color_buffer = nullptr;
  Normal* normal_buffer = nullptr;
  float* depth_buffer = nullptr;
  std::uint8_t* bounces_left_buffer = nullptr;

  PathsView() = default;
  PathsView(Paths& paths, unsigned int paths_count)
      : paths_count{paths_count}, rays{paths.rays.data()},
        pixel_indices{paths.pixel_indices.data()},
        color_buffer{paths.color_buffer.data()},
        normal_buffer{paths.normal_buffer.data()},
        depth_buffer{paths.depth_buffer.data()},
        bounces_left_buffer{paths.bounces_left_buffer.data()}
  {
  }
};

enum class GPUMethod { megakernel, wavefront };
inline constexpr const char* gpu_method_names[] = {"Megakernel", "Wavefront"};

class PathTracer {
public:
  int max_iterations = 1;

  GPUMethod current_gpu_method = GPUMethod::megakernel;

  EdgeAvoidingATrousDenoiser atrous_denoiser{};

private:
  Scene dev_scene_;

  Paths paths_;

  cuda::Buffer<glm::vec3> dev_color_buffer_;
  cuda::Buffer<Normal> dev_normal_buffer_;
  cuda::Buffer<float> dev_depth_buffer_;

  cuda::Buffer<Intersection> dev_intersection_buffer_;

  cuda::Buffer<glm::vec3> dev_denoised_buffer_;
  cuda::Buffer<glm::vec3> dev_denoised_buffer2_;
  glm::vec3* path_trace_result_buffer_ = nullptr;

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
