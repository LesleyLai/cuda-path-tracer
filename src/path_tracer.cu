#include "path_tracer.hpp"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <fmt/format.h>

#include <glm/gtx/compatibility.hpp>

#include "sphere.hpp"

void check_CUDA_error(std::string_view msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fmt::print(stderr, "Cuda error: {}: {}.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

struct Index2D {
  unsigned int x = 0;
  unsigned int y = 0;
};

[[nodiscard]] __device__ auto calculate_index_2d()
{
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  return Index2D{x, y};
}

__global__ void raygen_kernel(Ray* rays, unsigned int width,
                              unsigned int height)
{
  const auto [x, y] = calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = x + (y * width);

  const float aspect_ratio =
      static_cast<float>(width) / static_cast<float>(height);

  const float viewport_height = 2.0f;
  const float viewport_width = aspect_ratio * viewport_height;
  const float focal_length = 1.0;

  const auto origin = glm::vec3(0, 0, 0);
  const auto horizontal = glm::vec3(viewport_width, 0, 0);
  const auto vertical = glm::vec3(0, viewport_height, 0);
  const auto lower_left_corner = origin - horizontal / 2.f - vertical / 2.f -
                                 glm::vec3(0, 0, focal_length);

  const auto u = static_cast<float>(x) / static_cast<float>(width - 1);
  const auto v = static_cast<float>(y) / static_cast<float>(height - 1);
  rays[index] =
      Ray{origin, lower_left_corner + u * horizontal + v * vertical - origin};
}

__device__ auto get_background_color(Ray r) -> glm::vec3
{
  const glm::vec3 unit_direction = glm::normalize(r.direction);
  const auto t = 0.5f * (unit_direction.y + 1.0f);
  return glm::lerp(glm::vec3(0.5, 0.7, 1.0), glm::vec3(1.0, 1.0, 1.0), t);
}

__global__ void create_visualization_kernel(uchar4* pbo, Ray* rays,
                                            unsigned int width,
                                            unsigned int height)
{
  const auto [x, y] = calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = x + (y * width);

  const auto ray = rays[index];

  float t = 0;
  glm::vec3 point = {};
  glm::vec3 normal = {};

  const glm::vec3 color =
      ray_sphere_intersection_test(ray, glm::vec3{0.0f, 0.0f, -1.0f}, 0.5f, t,
                                   point, normal)
          ? ((normal + 1.0f) * 0.5f)
          : get_background_color(ray);

  constexpr auto normalize_color = [](float v) {
    return static_cast<char>(v * 255.99f);
  };

  if (x <= width && y <= height) {
    pbo[index].w = 1;
    pbo[index].x = normalize_color(color.x);
    pbo[index].y = normalize_color(color.y);
    pbo[index].z = normalize_color(color.z);
  }
}

PathTracer::PathTracer() = default;

void PathTracer::path_trace(uchar4* PBOpos, unsigned int width,
                            unsigned int height)
{
  constexpr unsigned int block_size = 16;
  const dim3 threads_per_block(block_size, block_size);

  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  raygen_kernel<<<full_blocks_per_grid, threads_per_block>>>(rays_.data(),
                                                             width, height);
  check_CUDA_error("Raygen Kernel");

  create_visualization_kernel<<<full_blocks_per_grid, threads_per_block>>>(
      PBOpos, rays_.data(), width, height);
  check_CUDA_error("Visualization kernel");

  CUDA_CHECK(cudaDeviceSynchronize());
}

void PathTracer::create_buffers(unsigned int width, unsigned int height)
{
  const auto pixel_count = static_cast<const std::size_t>(width) * height;
  rays_ = cuda::make_buffer<Ray>(pixel_count);
  CUDA_CHECK(cudaDeviceSynchronize());
}