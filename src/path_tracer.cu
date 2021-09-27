#include "path_tracer.hpp"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <fmt/format.h>

#include <glm/gtx/compatibility.hpp>

void check_CUDA_error(std::string_view msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fmt::print(stderr, "Cuda error: {}: {}.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(ans)                                                        \
  {                                                                            \
    cuda_check_impl((ans), __FILE__, __LINE__);                                \
  }
void cuda_check_impl(cudaError_t code, const char* file, int line,
                     bool abort = true)
{
  if (code != cudaSuccess) {
    fmt::print(stderr, "CUDA error: {} {} {}\n", cudaGetErrorString(code), file,
               line);
    if (abort) exit(code);
  }
}

__global__ void raygen_kernel(Ray* rays, unsigned int width,
                              unsigned int height)
{
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
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
  glm::vec3 unit_direction = glm::normalize(r.direction);
  auto t = 0.5f * (unit_direction.y + 1.0f);
  return glm::lerp(glm::vec3(0.5, 0.7, 1.0), glm::vec3(1.0, 1.0, 1.0), t);
}

__global__ void create_visualization_kernel(uchar4* pbo, Ray* rays,
                                            float time_since_start_s,
                                            unsigned int width,
                                            unsigned int height)
{
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto index = x + (y * width);

  const glm::vec3 background_color = get_background_color(rays[index]);

  constexpr auto normalize_color = [](float v) {
    return static_cast<char>(v * 255.99f);
  };

  if (x <= width && y <= height) {
    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 1;
    pbo[index].x = normalize_color(background_color.x);
    pbo[index].y = normalize_color(background_color.y);
    pbo[index].z = normalize_color(background_color.z);
  }
}

PathTracer::PathTracer() = default;

PathTracer::~PathTracer()
{
  cudaFree(rays_);
}

void PathTracer::path_trace(uchar4* PBOpos, unsigned int width,
                            unsigned int height)
{
  // set up crucial magic
  constexpr unsigned int block_size = 16;
  const dim3 threads_per_block(block_size, block_size);

  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  raygen_kernel<<<full_blocks_per_grid, threads_per_block>>>(rays_, width,
                                                             height);
  check_CUDA_error("Raygen Kernel");

  create_visualization_kernel<<<full_blocks_per_grid, threads_per_block>>>(
      PBOpos, rays_, 0, width, height);

  cudaDeviceSynchronize();

  check_CUDA_error("Visualization kernel");
}

void PathTracer::create_buffers(unsigned int width, unsigned int height)
{
  const std::size_t pixel_count = width * height;
  CUDA_CHECK(cudaMalloc(&rays_, pixel_count * sizeof(Ray)));
  CUDA_CHECK(cudaDeviceSynchronize());
}