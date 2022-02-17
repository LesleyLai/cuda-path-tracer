#include "edge_avoiding_a_trous_denoiser.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <tuple>

#include "../cuda_utils/2d_indices.cuh"
#include "../cuda_utils/cuda_check.hpp"

__global__ void
denoising_kernel(unsigned int width, unsigned int height,
                 EdgeAvoidingATrousDenoiser::Parameters parameters,
                 const glm::vec3* color_buffer, const glm::vec3* normal_buffer,
                 const glm::vec3* position_buffer, glm::vec3* out_buffer,
                 int step_width)
{
  const auto [x, y] = cuda::calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = FLATTERN_INDEX(x, y);

  const float c_phi = parameters.color_weight;
  const float n_phi = parameters.normal_weight;
  const float p_phi = parameters.position_weight;

  // 5x5 symmetric kernel
  constexpr float kernel[] = {3.f / 8.f, 1.f / 4.f, 1.f / 16.f};

  const glm::vec3 cval = color_buffer[index];
  const glm::vec3 nval = normal_buffer[index];
  const glm::vec3 pval = position_buffer[index];

  glm::vec3 sum{0.0};
  float cum_w = 0.0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      const int u = std::clamp(static_cast<int>(x) + dx * step_width, 0,
                               static_cast<int>(width));
      const int v = std::clamp(static_cast<int>(y) + dy * step_width, 0,
                               static_cast<int>(height));
      const auto temp_index = FLATTERN_INDEX(u, v);

      const glm::vec3 ctemp = color_buffer[temp_index];
      glm::vec3 t = cval - ctemp;
      float dist2 = glm::dot(t, t);
      const float c_w = std::min(std::exp(-dist2 / c_phi), 1.0f);

      const glm::vec3 ntemp = normal_buffer[temp_index];
      t = nval - ntemp;
      dist2 = std::max(
          glm::dot(t, t) / static_cast<float>(step_width * step_width), 0.0f);
      const float n_w = std::min(std::exp(-dist2 / n_phi), 1.0f);

      const glm::vec3 ptmp = position_buffer[temp_index];
      t = pval - ptmp;
      dist2 = glm::dot(t, t);
      const float p_w = std::min(std::exp(-dist2 / p_phi), 1.0f);

      const float weight = c_w * n_w * p_w;

      const int kernel_index = std::min(std::abs(dx), std::abs(dy));
      sum += ctemp * weight * kernel[kernel_index];
      cum_w += weight * kernel[kernel_index];
    }
  }

  out_buffer[index] = sum / cum_w;
}

auto EdgeAvoidingATrousDenoiser::denoise(
    unsigned int width, unsigned int height, const glm::vec3* color_buffer,
    const glm::vec3* normal_buffer, const glm::vec3* position_buffer,
    glm::vec3* back_buffer, glm::vec3* front_buffer) -> glm::vec3*
{
  constexpr unsigned int block_size = 16;
  const dim3 threads_per_block(block_size, block_size);

  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  Parameters parameters_copy = parameters;
  for (int step_width = 1; step_width <= parameters_copy.filter_size;
       step_width *= 2) {
    denoising_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        width, height, parameters_copy, color_buffer, normal_buffer,
        position_buffer, back_buffer, step_width);
    std::tie(color_buffer, back_buffer, front_buffer) =
        std::tie(back_buffer, front_buffer, back_buffer);
    parameters_copy.color_weight /= 2;
  }
  cuda::check_CUDA_error("Denoising kernel");
  return front_buffer;
}