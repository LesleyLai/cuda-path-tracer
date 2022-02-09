#ifndef CUDA_PATH_TRACER_EDGE_AVOIDING_A_TROUS_DENOISER_HPP
#define CUDA_PATH_TRACER_EDGE_AVOIDING_A_TROUS_DENOISER_HPP

#include <glm/vec3.hpp>

struct ATrousParameters {
  int filter_size = 10;
  float color_weight = 0.45f;
  float normal_weight = 0.30f;
  float position_weight = 0.25f;
};

class EdgeAvoidingATrousDenoiser {
public:
  ATrousParameters parameters;

  /**
   * @return A pointer to the buffer of denoising output
   */
  [[nodiscard]] auto denoise(unsigned int width, unsigned int height,
                             const glm::vec3* color_buffer,
                             const glm::vec3* normal_buffer,
                             const glm::vec3* position_buffer,
                             glm::vec3* buffer1, glm::vec3* buffer2)
      -> glm::vec3*;
};

#endif // CUDA_PATH_TRACER_EDGE_AVOIDING_A_TROUS_DENOISER_HPP
