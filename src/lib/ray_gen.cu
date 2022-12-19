#include "ray_gen.cuh"

#include "constant_memory.cuh"
#include "cuda_utils/cuda_check.hpp"
#include "cuda_utils/indices.cuh"
#include "hash.cuh"
#include "path_tracer.hpp"

#include <thrust/random.h>

__global__ void raygen_kernel(std::size_t iteration, PathsView paths)
{
  GPUCamera camera = constant_memory::gpu_camera;
  const auto [x, y] = cuda::calculate_index_2d();
  if (x >= camera.width || y >= camera.height) return;

  const auto index = cuda::flattern_index({x, y}, camera.width, camera.height);
  thrust::default_random_engine rng(hash(hash(index) ^ iteration));

  thrust::uniform_real_distribution<float> dist(0.0, 1.0);
  const auto fx = static_cast<float>(x) + dist(rng);
  const auto fy = static_cast<float>(y) + dist(rng);

  auto ray = generate_ray(camera, fx, fy);

  paths.color_buffer[index] = glm::vec3{1.0, 1.0, 1.0};
  paths.depth_buffer[index] = 1e6;
  paths.normal_buffer[index] = -ray.direction;
  paths.bounces_left_buffer[index] = 50;
  paths.rays[index] = ray;
  paths.pixel_indices[index] = static_cast<int>(index);
}

[[nodiscard]] __device__ auto generate_ray(const GPUCamera& camera, float x,
                                           float y) -> Ray
{
  const float aspect_ratio =
      static_cast<float>(camera.width) / static_cast<float>(camera.height);

  const float viewport_height = 2.0f * tan(camera.vfov / 2);
  const float viewport_width = aspect_ratio * viewport_height;
  const float focal_length = 1.0;

  const auto origin = glm::vec3(0, 0, 0);
  const auto horizontal = glm::vec3(viewport_width, 0, 0);
  const auto vertical = glm::vec3(0, viewport_height, 0);
  const auto lower_left_corner = origin - horizontal / 2.f - vertical / 2.f -
                                 glm::vec3(0, 0, focal_length);

  const auto u = x / static_cast<float>(camera.width - 1);
  const auto v = (static_cast<float>(camera.height) - y) /
                 static_cast<float>(camera.height - 1);
  const auto direction =
      lower_left_corner + u * horizontal + v * vertical - origin;

  const auto world_origin =
      glm::vec3(camera.camera_matrix * glm::vec4(origin, 1.0));
  const auto world_direction = glm::normalize(
      glm::vec3(camera.camera_matrix * glm::vec4(direction, 0.0)));
  return Ray{world_origin, 1e-4, world_direction, FLT_MAX};
}

void generate_rays(unsigned int iteration, const Camera& camera,
                   UResolution resolution, PathsView paths)
{
  const auto [width, height] = resolution;

  const dim3 block_size(8, 8);
  const dim3 blocks_per_grid( //
      (width + block_size.x - 1) / block_size.x,
      (height + block_size.y - 1) / block_size.y);

  const auto gpu_camera = camera.to_gpu_camera(resolution);
  cudaMemcpyToSymbol(constant_memory::gpu_camera, &gpu_camera,
                     sizeof(GPUCamera));

  raygen_kernel<<<blocks_per_grid, block_size>>>(iteration, paths);
  cuda::check_CUDA_error("Raygen kernel");
}