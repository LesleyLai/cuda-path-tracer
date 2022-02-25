#ifndef CUDA_PATH_TRACER_EDGE_AVOIDING_A_TROUS_DENOISER_HPP
#define CUDA_PATH_TRACER_EDGE_AVOIDING_A_TROUS_DENOISER_HPP

#include "../camera.hpp"
#include <glm/glm.hpp>

class EdgeAvoidingATrousDenoiser {
public:
  int filter_size = 10;
  float color_weight = 0.45f;
  float normal_weight = 0.30f;
  float position_weight = 0.25f;

  /**
   * @return A pointer to the buffer of denoising output
   */
  [[nodiscard]] auto denoise(unsigned int width, unsigned int height,
                             const glm::vec3* color_buffer,
                             const glm::vec3* normal_buffer,
                             const float* depth_buffer, glm::vec3* back_buffer,
                             glm::vec3* front_buffer) -> glm::vec3*;
};

#endif // CUDA_PATH_TRACER_EDGE_AVOIDING_A_TROUS_DENOISER_HPP
