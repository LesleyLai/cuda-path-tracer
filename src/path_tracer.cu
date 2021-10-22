#include "path_tracer.hpp"

#include "camera.hpp"
#include "distributions.cuh"
#include "span.hpp"

#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust/random.h>

#include <cmath>
#include <fmt/format.h>

#include <iterator>

#include <glm/gtx/compatibility.hpp>

static const Sphere spheres[] = {
    {{0.0f, -100.5f, -1.0f}, 100.f, 0},
    {{0.0f, 0.0f, -1.0f}, 0.5f, 1},
    {{-1.0f, 0.0f, -1.0f}, 0.5f, 2},
    {{1.0f, 0.0f, -1.0f}, 0.5f, 3},
};

static constexpr Material mat[] = {{Material::Type::Diffuse, 0},
                                   {Material::Type::Diffuse, 1},
                                   {Material::Type::Dielectric, 0},
                                   {Material::Type::Metal, 0}};

static const DiffuseMateral diffuse_mat[] = {{{0.8, 0.8, 0.0}},
                                             {{0.1, 0.2, 0.5}}};
static const MetalMaterial metal_mat[] = {{{0.8, 0.6, 0.2}, 1.0}};
static const DielectricMaterial dielectric_mat[] = {{1.5}};

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

[[nodiscard]] __device__ auto raygen(glm::mat4 camera_matrix,
                                     unsigned int width, unsigned int height,
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
  const auto direction =
      lower_left_corner + u * horizontal + v * vertical - origin;

  const auto world_origin = glm::vec3(camera_matrix * glm::vec4(origin, 1.0));
  const auto world_direction =
      glm::normalize(glm::vec3(camera_matrix * glm::vec4(direction, 0.0)));
  return Ray{world_origin, world_direction};
}

__device__ auto get_background_color(Ray r) -> glm::vec3
{
  const glm::vec3 unit_direction = glm::normalize(r.direction);
  const auto t = 0.5f * (unit_direction.y + 1.0f);
  return glm::lerp(glm::vec3(0.5, 0.7, 1.0), glm::vec3(1.0, 1.0, 1.0), t);
}

__device__ auto ray_scene_intersection_test(Ray ray, Span<const Sphere> spheres,
                                            float t_min, float t_max,
                                            HitRecord& record) -> bool
{
  bool hit = false;
  for (const auto& sphere : spheres) {
    HitRecord new_record;
    if (ray_sphere_intersection_test(ray, sphere, new_record)) {
      if (new_record.t <= t_max && new_record.t >= t_min) {
        hit = true;
        record = new_record;
        t_max = new_record.t;
      }
    }
  }
  return hit;
}

[[nodiscard]] __host__ __device__ constexpr auto hash(unsigned int a)
    -> unsigned int
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__device__ static auto reflectance(float cosine, float ref_idx) -> float
{
  // Use Schlick's approximation for reflectance.
  auto r0 = (1 - ref_idx) / (1 + ref_idx);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__global__ void path_tracing_kernel(
    glm::mat4 camera_matrix, uchar4* pbo, glm::vec3* image,
    std::size_t iteration, Span<const Sphere> spheres, Span<const Material> mat,
    Span<const DiffuseMateral> diffuse_mat, Span<const MetalMaterial> metal_mat,
    Span<const DielectricMaterial> dielectric_mat, unsigned int width,
    unsigned int height)
{
  const auto [x, y] = calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = x + ((height - y) * width);

  thrust::default_random_engine rng(hash(hash(index) ^ iteration));

  // ray gen
  auto ray = raygen(camera_matrix, width, height, x, y, rng);

  // Path tracing
  glm::vec3 color{1.0f, 1.0f, 1.0f};
  for (int i = 0; i < 50; ++i) {
    HitRecord record;
    float t_max = std::numeric_limits<float>::max();
    const bool hit =
        ray_scene_intersection_test(ray, spheres, 1e-5f, t_max, record);
    if (!hit) {
      color *= get_background_color(ray);
      break;
    }
    ray.origin =
        record.point -
        1e-4f * glm::sign(dot(ray.direction, record.normal)) * record.normal;
    // material stuff
    const Material& material = mat[record.material_id];
    switch (material.type) {
    case Material::Type::Diffuse: {
      auto scatter_direction =
          glm::normalize(record.normal + random_in_unit_sphere(rng));

      // Catch degenerated case
      if (abs(scatter_direction.x) < 1e-8 && abs(scatter_direction.y) < 1e-8 &&
          abs(scatter_direction.z) < 1e-8) {
        scatter_direction = record.normal;
      }

      ray.direction = scatter_direction;
      color *= diffuse_mat[material.index].albedo;
    } break;
    case Material::Type::Metal: {
      const auto metal = metal_mat[material.index];
      const auto reflected = glm::reflect(ray.direction, record.normal);
      const auto scatter_direction =
          reflected + metal.fuzz * random_in_unit_sphere(rng);
      ray.direction = scatter_direction;
      if (dot(scatter_direction, record.normal) > 0) {
        color *= metal.albedo;
      } else {
        color = glm::vec3(0.0, 0.0, 0.0);
      }
    } break;
    case Material::Type::Dielectric: {
      const auto dielectric = dielectric_mat[material.index];
      const auto refraction_ratio = record.side == HitFaceSide::front
                                        ? (1.0f / dielectric.refraction_index)
                                        : dielectric.refraction_index;

      const auto unit_direction = normalize(ray.direction);
      const float cos_theta = min(dot(-unit_direction, record.normal), 1.0f);
      const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

      const bool cannot_refract = refraction_ratio * sin_theta > 1.0;
      thrust::uniform_real_distribution<float> dist(0.0, 1.0);
      const glm::vec3 direction = [&]() {
        if (cannot_refract ||
            reflectance(cos_theta, refraction_ratio) > dist(rng)) {
          return reflect(unit_direction, record.normal);
        } else {
          return refract(unit_direction, record.normal, refraction_ratio);
        }
      }();

      ray = Ray{record.point, direction};
    } break;
    }
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

void PathTracer::path_trace(uchar4* dev_pbo, const Camera& camera,
                            unsigned int width, unsigned int height)
{
  if (iteration_ >= max_iterations) return;

  constexpr unsigned int block_size = 16;
  const dim3 threads_per_block(block_size, block_size);

  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  path_tracing_kernel<<<full_blocks_per_grid, threads_per_block>>>(
      camera.camera_matrix(), dev_pbo, dev_image_.data(), iteration_,
      Span{dev_spheres_.data(), std::size(spheres)},
      Span{dev_mat_.data(), std::size(mat)},
      Span{dev_diffuse_mat_.data(), std::size(diffuse_mat)},
      Span{dev_metal_mat_.data(), std::size(metal_mat)},
      Span{dev_dielectric_mat_.data(), std::size(dielectric_mat)}, width,
      height);
  check_CUDA_error("Visualization kernel");

  CUDA_CHECK(cudaDeviceSynchronize());

  ++iteration_;
}

void PathTracer::restart()
{
  iteration_ = 0;
}

void PathTracer::resize_image(unsigned int width, unsigned int height)
{
  iteration_ = 0;
  dev_image_ = cuda::make_buffer<glm::vec3>(width * height);
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
[[nodiscard]] auto create_buffer_from_cpu_data(Span<const T> span)
{
  auto dev_buffer = cuda::make_buffer<T>(span.size());
  CUDA_CHECK(cudaMemcpy(dev_buffer.data(), span.data(), span.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return dev_buffer;
}

void PathTracer::create_buffers(unsigned int width, unsigned int height)
{
  dev_spheres_ = create_buffer_from_cpu_data(Span{spheres});
  dev_mat_ = create_buffer_from_cpu_data(Span{mat});
  dev_diffuse_mat_ = create_buffer_from_cpu_data(Span{diffuse_mat});
  dev_metal_mat_ = create_buffer_from_cpu_data(Span{metal_mat});
  dev_dielectric_mat_ = create_buffer_from_cpu_data(Span{dielectric_mat});
  resize_image(width, height);
}
