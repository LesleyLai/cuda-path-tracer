#include "edge_avoiding_a_trous_denoiser.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <tuple>

#include "../constant_memory.cuh"
#include "../cuda_utils/2d_indices.cuh"
#include "../cuda_utils/cuda_check.hpp"
#include "../ray_gen.cuh"

namespace constant_memory {

namespace {

__constant__ float color_weight;
__constant__ float normal_weight;
__constant__ float position_weight;

} // anonymous namespace

} // namespace constant_memory

__global__ void denoising_kernel(const glm::vec3* color_buffer,
                                 const glm::vec3* normal_buffer,
                                 const float* depth_buffer,
                                 glm::vec3* out_buffer, int step_width)
{
  const auto camera = constant_memory::gpu_camera;
  const auto [x, y] = cuda::calculate_index_2d();
  if (x >= camera.width || y >= camera.height) return;
  const auto index = cuda::flattern_index({x, y}, camera.width, camera.height);

  const float c_phi = constant_memory::color_weight;
  const float n_phi = constant_memory::normal_weight;
  const float p_phi = constant_memory::position_weight;

  // 5x5 symmetric kernel
  static constexpr float kernel[] = {3.f / 8.f, 1.f / 4.f, 1.f / 16.f};

  const glm::vec3 cval = color_buffer[index];
  const glm::vec3 nval = normal_buffer[index];

  const auto ray = generate_ray(camera, x + 0.5f, y + 0.5f);
  const glm::vec3 pval = ray(depth_buffer[index]);

  glm::vec3 sum{0.0};
  float cum_w = 0.0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      const int u = std::clamp(static_cast<int>(x) + dx * step_width, 0,
                               static_cast<int>(camera.width));
      const int v = std::clamp(static_cast<int>(y) + dy * step_width, 0,
                               static_cast<int>(camera.height));
      const auto temp_index =
          cuda::flattern_index(cuda::Index2D{static_cast<unsigned int>(u),
                                             static_cast<unsigned int>(v)},
                               camera.width, camera.height);

      const glm::vec3 ctemp = color_buffer[temp_index];
      glm::vec3 t = cval - ctemp;
      float dist2 = glm::dot(t, t);
      const float c_w = std::min(std::exp(-dist2 / c_phi), 1.0f);

      const glm::vec3 ntemp = normal_buffer[temp_index];
      t = nval - ntemp;
      dist2 = std::max(
          glm::dot(t, t) / static_cast<float>(step_width * step_width), 0.0f);
      const float n_w = std::min(std::exp(-dist2 / n_phi), 1.0f);

      const auto temp_ray = generate_ray(camera, u + 0.5f, v + 0.5f);
      const glm::vec3 ptmp = temp_ray(depth_buffer[temp_index]);
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
    const glm::vec3* normal_buffer, const float* depth_buffer,
    glm::vec3* back_buffer, glm::vec3* front_buffer) -> glm::vec3*
{
  constexpr unsigned int block_size = 16;
  const dim3 threads_per_block(block_size, block_size);

  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  cudaMemcpyToSymbol(constant_memory::color_weight, &color_weight,
                     sizeof(float));
  cudaMemcpyToSymbol(constant_memory::normal_weight, &normal_weight,
                     sizeof(float));
  cudaMemcpyToSymbol(constant_memory::position_weight, &position_weight,
                     sizeof(float));

  for (int step_width = 1; step_width <= filter_size; step_width *= 2) {
    denoising_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        color_buffer, normal_buffer, depth_buffer, back_buffer, step_width);
    std::tie(color_buffer, back_buffer, front_buffer) =
        std::tie(back_buffer, front_buffer, back_buffer);
  }
  cuda::check_CUDA_error("Denoising kernel");

  return front_buffer;
}