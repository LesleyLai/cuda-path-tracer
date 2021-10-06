#include "path_tracer.hpp"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust/random.h>

#include <cmath>
#include <fmt/format.h>

#include <glm/gtx/compatibility.hpp>

static const Sphere spheres[] = {{{0.0f, 0.0f, -1.0f}, 0.5f},
                                 {{0.0f, -100.5f, -1.0f}, 100.f}};

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

[[nodiscard]] __device__ auto calculate_index_2d() -> Index2D
{
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  return Index2D{x, y};
}

[[nodiscard]] __device__ auto raygen(unsigned int width, unsigned int height,
                                     unsigned int x, unsigned int y,
                                     thrust::default_random_engine& rng) -> Ray
{
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

  thrust::uniform_real_distribution<float> dist(0.0, 1.0);

  const auto u =
      (static_cast<float>(x) + dist(rng)) / static_cast<float>(width - 1);
  const auto v =
      (static_cast<float>(y) + dist(rng)) / static_cast<float>(height - 1);
  return Ray{origin,
             lower_left_corner + u * horizontal + v * vertical - origin};
}

__device__ auto get_background_color(Ray r) -> glm::vec3
{
  const glm::vec3 unit_direction = glm::normalize(r.direction);
  const auto t = 0.5f * (unit_direction.y + 1.0f);
  return glm::lerp(glm::vec3(0.5, 0.7, 1.0), glm::vec3(1.0, 1.0, 1.0), t);
}

__device__ auto ray_scene_intersection_test(Ray ray, Sphere* spheres,
                                            std::size_t sphere_count,
                                            HitRecord& record) -> bool
{
  bool hit = false;
  float t_max = std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < sphere_count; ++i) {
    const auto& sphere = spheres[i];
    HitRecord new_record;
    if (ray_sphere_intersection_test(ray, sphere.center, sphere.radius,
                                     new_record)) {
      hit = true;
      if (new_record.t < t_max) {
        record = new_record;
        t_max = new_record.t;
      }
    }
  }
  return hit;
}

[[nodiscard]] __device__ auto
random_in_unit_sphere(thrust::default_random_engine& rng) -> glm::vec3
{
  thrust::uniform_real_distribution<float> uni(-1, 1);
  thrust::normal_distribution<float> normal(0, 1);

  glm::vec3 p{normal(rng), normal(rng), normal(rng)};
  p = normalize(p);

  const auto c = std::cbrt(uni(rng));
  return p * c;
}

__host__ __device__ constexpr unsigned int hash(unsigned int a)
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__global__ void path_tracing_kernel(uchar4* pbo, glm::vec3* image,
                                    std::size_t iteration, Sphere* spheres,
                                    std::size_t sphere_count,
                                    unsigned int width, unsigned int height)
{
  const auto [x, y] = calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = x + ((height - y) * width);

  thrust::default_random_engine rng(hash(hash(index) ^ iteration));

  // ray gen
  auto ray = raygen(width, height, x, y, rng);

  // Path tracing
  glm::vec3 color{1.0f, 1.0f, 1.0f};

  for (int i = 0; i < 50; ++i) {
    HitRecord record;
    const bool hit =
        ray_scene_intersection_test(ray, spheres, sphere_count, record);
    if (!hit) {
      color *= get_background_color(ray);
      break;
    }
    ray.origin =
        record.point -
        0.0001f * glm::sign(dot(ray.direction, record.normal)) * record.normal;
    ray.direction = glm::normalize(record.normal + random_in_unit_sphere(rng));
    color *= 0.5f;
  }
  // gamma correction
  color.x = glm::pow(color.x, 1.f / 2.2f);
  color.y = glm::pow(color.y, 1.f / 2.2f);
  color.z = glm::pow(color.z, 1.f / 2.2f);

  // Final gathering
  const auto sample_count = static_cast<float>(iteration + 1);
  image[index] = (image[index] * (sample_count - 1) + color) / sample_count;

  // Visualization
  constexpr auto normalize_color = [](float v) {
    return static_cast<unsigned char>(glm::clamp(v, 0.f, 1.f) * 255.99f);
  };
  if (x <= width && y <= height) {
    pbo[index].w = 1;
    pbo[index].x = normalize_color(image[index].x);
    pbo[index].y = normalize_color(image[index].y);
    pbo[index].z = normalize_color(image[index].z);
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

  path_tracing_kernel<<<full_blocks_per_grid, threads_per_block>>>(
      PBOpos, dev_image_.data(), iteration_, dev_spheres_.data(),
      std::size(spheres), width, height);
  check_CUDA_error("Visualization kernel");

  CUDA_CHECK(cudaDeviceSynchronize());

  ++iteration_;
}

void PathTracer::reset()
{
  iteration_ = 0;
}

void PathTracer::resize_image(unsigned int width, unsigned int height)
{
  iteration_ = 0;
  dev_image_ = cuda::make_buffer<glm::vec3>(width * height);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void PathTracer::create_buffers(unsigned int width, unsigned int height)
{
  dev_spheres_ = cuda::make_buffer<Sphere>(std::size(spheres));
  CUDA_CHECK(cudaMemcpy(dev_spheres_.data(), spheres,
                        std::size(spheres) * sizeof(Sphere),
                        cudaMemcpyHostToDevice));
  resize_image(width, height);
}