#include "kernel.hpp"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <fmt/format.h>

void check_CUDA_error(std::string_view msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fmt::print(stderr, "Cuda error: {}: {}.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void kern_create_visualization(uchar4* pbo, float time_since_start_s,
                                          unsigned int width,
                                          unsigned int height)
{
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto index = x + (y * width);

  const float fx = static_cast<float>(x) / static_cast<float>(width);
  const float fy = static_cast<float>(y) / static_cast<float>(height);

  constexpr auto normalize_trig_value = [](float v) {
    return static_cast<char>((v + 1.0f) / 2.0f * 255.99f);
  };

  if (x <= width && y <= height) {
    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 1;
    pbo[index].x = normalize_trig_value(std::sin(fx + time_since_start_s));
    pbo[index].y = normalize_trig_value(std::cos(fy + time_since_start_s));
    pbo[index].z = 0;
  }
}

void execute_kernel(uchar4* PBOpos, float time_since_start_s,
                    unsigned int width, unsigned int height)
{
  // set up crucial magic
  constexpr unsigned int block_size = 16;
  const dim3 threads_per_block(block_size, block_size);

  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  kern_create_visualization<<<full_blocks_per_grid, threads_per_block>>>(
      PBOpos, time_since_start_s, width, height);

  cudaDeviceSynchronize();

  check_CUDA_error("Kernel failed!");
}
