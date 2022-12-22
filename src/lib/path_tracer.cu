#include "path_tracer.hpp"

#include "constant_memory.cuh"
#include "cuda_utils/cuda_buffer.hpp"
#include "cuda_utils/indices.cuh"
#include "distributions.cuh"
#include "hash.cuh"
#include "ray_gen.cuh"
#include "span.hpp"
#include "static_stack.hpp"
#include "transform.hpp"

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust/random.h>
#include <thrust/sort.h>

#include <cmath>
#include <fmt/format.h>

#include <iterator>

#include <glm/gtx/compatibility.hpp>
#include <lib/accelerators/bvh.hpp>

#include "intersections.cuh"

static constexpr std::uint8_t max_bounces = 50;

__device__ auto get_background_color(Ray r) -> glm::vec3
{
  const glm::vec3 unit_direction = glm::normalize(r.direction);
  const auto t = 0.5f * (unit_direction.y + 1.0f);
  return glm::lerp(glm::vec3(0.5, 0.7, 1.0), glm::vec3(1.0, 1.0, 1.0), t);
}

__device__ auto ray_mesh_intersection_test(Ray ray, const glm::vec3* positions,
                                           Span<const std::uint32_t> indices,
                                           Span<const BVHNode> bvh,
                                           const Transform& transform,
                                           Intersection& record) -> bool
{
  bool hit = false;

  const Ray transformed_ray = inverse_transform_ray(transform, ray);

  StaticStack<unsigned int, 24> node_stack;
  node_stack.push(0);
  while (node_stack.size() != 0) {
    const std::uint32_t node_index = node_stack.pop();
    const BVHNode node = bvh[node_index];

    if (node.is_leaf) {
      const std::uint32_t i = node.data.leaf.triangle_index_begin;

      const std::uint32_t index0 = indices[i];
      const std::uint32_t index1 = indices[i + 1];
      const std::uint32_t index2 = indices[i + 2];
      const glm::vec3 p0 = transform_point(transform, positions[index0]);
      const glm::vec3 p1 = transform_point(transform, positions[index1]);
      const glm::vec3 p2 = transform_point(transform, positions[index2]);

      if (ray_triangle_intersection_test(ray, p0, p1, p2, record)) {
        hit = true;
        ray.t_max = record.t;
      }
    } else {
      // Intersect AABB for an inner node
      if (ray_aabb_intersection_test(transformed_ray, node.aabb)) {
        const auto [left_index, right_index] = node.data.inner;
        node_stack.push(right_index);
        node_stack.push(left_index);
      }
    }
  }

  // for (std::size_t j = 0; j < indices.size(); j += 3) {
  //   const auto index0 = indices[j];
  //   const auto index1 = indices[j + 1];
  //   const auto index2 = indices[j + 2];
  //
  //   const auto p0 = transform_point(transform, positions[index0]);
  //   const auto p1 = transform_point(transform, positions[index1]);
  //   const auto p2 = transform_point(transform, positions[index2]);
  //
  //   if (ray_triangle_intersection_test(ray, p0, p1, p2, record)) {
  //     hit = true;
  //     ray.t_max = record.t;
  //   }
  // }
  return hit;
}

__device__ auto ray_object_intersection_test(Ray ray, GPUObject obj,
                                             AggregateView aggregate,
                                             Intersection& record) -> bool
{
  bool hit = false;

  if (!ray_aabb_intersection_test(ray, obj.aabb)) { return false; }

  switch (obj.type) {
  case ObjectType::sphere: {
    const auto transformed_ray = inverse_transform_ray(obj.transform, ray);
    const auto sphere = aggregate.spheres[obj.index];
    hit = ray_sphere_intersection_test(transformed_ray, sphere, record);

    if (hit) {
      record.point = transform_point(obj.transform, record.point);
      record.t = glm::distance(ray.origin, record.point);
      record.normal = transform_normal(obj.transform, record.normal);
    }

    break;
  }
  case ObjectType::mesh:
    hit =
        ray_mesh_intersection_test(ray, aggregate.positions, aggregate.indices,
                                   aggregate.bvh, obj.transform, record);
    break;
  }

  return hit;
}

__device__ auto ray_scene_intersection_test(Ray ray, AggregateView aggregate,
                                            Intersection& record) -> bool
{
  bool hit = false;

  const auto objects = aggregate.objects;
  const auto* object_material_indices = aggregate.object_material_indices;

  for (std::size_t i = 0; i < objects.size(); ++i) {
    const GPUObject obj = objects[i];
    if (ray_object_intersection_test(ray, obj, aggregate, record)) {
      hit = true;
      record.material_id = object_material_indices[i];
      ray.t_max = record.t;
    }
  }

  return hit;
}

__device__ static auto reflectance(float cosine, float ref_idx) -> float
{
  // Use Schlick's approximation for reflectance.
  auto r0 = (1 - ref_idx) / (1 + ref_idx);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ void evaluate_material(Ray& ray, const Intersection intersection,
                                  thrust::default_random_engine& rng,
                                  glm::vec3& color, const Material* materials)
{
  ray.origin = intersection.point -
               1e-4f * glm::sign(dot(ray.direction, intersection.normal)) *
                   intersection.normal;
  // material stuff
  const Material& material = materials[intersection.material_id];
  switch (material.type) {
  case Material::Type::Diffuse: {
    auto diffuse = material.data.diffuse;
    auto scatter_direction =
        glm::normalize(intersection.normal + random_in_unit_sphere(rng));

    // Catch degenerated case
    if (abs(scatter_direction.x) < 1e-8 && abs(scatter_direction.y) < 1e-8 &&
        abs(scatter_direction.z) < 1e-8) {
      scatter_direction = intersection.normal;
    }

    ray.direction = scatter_direction;
    color *= diffuse.albedo;
  } break;
  case Material::Type::Metal: {
    const auto metal = material.data.metal;
    const auto reflected = glm::reflect(ray.direction, intersection.normal);
    const auto scatter_direction =
        reflected + metal.fuzz * random_in_unit_sphere(rng);
    ray.direction = scatter_direction;
    if (dot(scatter_direction, intersection.normal) > 0) {
      color *= metal.albedo;
    } else {
      color = glm::vec3(0.0, 0.0, 0.0);
    }
  } break;
  case Material::Type::Dielectric: {
    const auto dielectric = material.data.dielectric;
    const auto refraction_ratio = intersection.side == HitFaceSide::front
                                      ? (1.0f / dielectric.refraction_index)
                                      : dielectric.refraction_index;

    const auto unit_direction = normalize(ray.direction);
    const float cos_theta =
        min(dot(-unit_direction, intersection.normal), 1.0f);
    const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    const bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    thrust::uniform_real_distribution<float> dist(0.0, 1.0);
    const glm::vec3 direction = [&]() {
      if (cannot_refract ||
          reflectance(cos_theta, refraction_ratio) > dist(rng)) {
        return reflect(unit_direction, intersection.normal);
      } else {
        return refract(unit_direction, intersection.normal, refraction_ratio);
      }
    }();

    ray = Ray{intersection.point, 1e-5, direction,
              std::numeric_limits<float>::max()};
  } break;
  }
}

__device__ void final_gather(std::size_t iteration, glm::vec3 new_color,
                             Normal new_normal, float new_depth,
                             glm::vec3& current_color, Normal& current_normal,
                             float& current_depth)
{
  const auto sample_count = static_cast<float>(iteration + 1);
  const auto temporal_accumulate = [sample_count, iteration](auto old_value,
                                                             auto new_value) {
    return iteration == 0
               ? new_value
               : (old_value * (sample_count - 1) + new_value) / sample_count;
  };

  current_color = temporal_accumulate(current_color, new_color);
  current_normal = temporal_accumulate(current_normal, new_normal);
  current_depth = temporal_accumulate(current_depth, new_depth);
}

[[nodiscard]] __device__ auto linear_to_gamma(glm::vec3 color) -> glm::vec3
{
  color = glm::pow(color, glm::vec3(1.f / 2.2f));
  return color;
}

__global__ void path_tracing_mega_kernel(const std::size_t iteration,
                                         const AggregateView aggregate,
                                         const Material* mat,
                                         glm::vec3* color_buffer,
                                         Normal* normal_buffer,
                                         float* depth_buffer)
{
  const auto camera = constant_memory::gpu_camera;
  const auto [x, y] = cuda::calculate_index_2d();
  if (x >= camera.width || y >= camera.height) return;

  const auto index = cuda::flattern_index({x, y}, camera.width, camera.height);
  thrust::default_random_engine rng(hash(hash(index) ^ iteration));

  thrust::uniform_real_distribution<float> dist(0.0, 1.0);
  const float fx = static_cast<float>(x) + dist(rng);
  const float fy = static_cast<float>(y) + dist(rng);

  auto ray = generate_ray(camera, fx, fy);

  // Path tracing
  glm::vec3 color{1.0f, 1.0f, 1.0f};
  glm::vec3 normal = -ray.direction;
  float depth = 1e6;

  for (int i = 0; i < max_bounces; ++i) {
    Intersection intersection;
    const bool hit = ray_scene_intersection_test(ray, aggregate, intersection);
    if (!hit) {
      color *= get_background_color(ray);
      break;
    }
    if (i == 0) {
      normal = intersection.normal;
      depth = intersection.t;
    }

    evaluate_material(ray, intersection, rng, color, mat);
  }

  final_gather(iteration, color, normal, depth, color_buffer[index],
               normal_buffer[index], depth_buffer[index]);
}

__global__ void intersection_kernel(PathsView paths,
                                    Intersection* intersections,
                                    AggregateView aggregate)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= paths.paths_count) return;

  const auto ray = paths.rays[index];

  Intersection intersection;
  const bool hit = ray_scene_intersection_test(ray, aggregate, intersection);

  if (hit) {
    intersections[index] = intersection;
  } else {
    // Negative t means no intersection
    intersections[index].t = -1.0;
    paths.bounces_left_buffer[index] = 0;
  }
}

__global__ void material_kernel(std::size_t iteration,
                                unsigned int current_bounce, PathsView paths,
                                const Material* mat,
                                const Intersection* intersections)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= paths.paths_count) return;

  thrust::default_random_engine rng(hash(hash(index) ^ iteration));
  rng.discard(current_bounce);

  const Intersection intersection = intersections[index];
  if (intersection.t < 0) {
    paths.color_buffer[index] *= get_background_color(paths.rays[index]);
    return;
  }
  if (current_bounce == 0) {
    paths.depth_buffer[index] = intersection.t;
    paths.normal_buffer[index] = intersection.normal;
  }

  evaluate_material(paths.rays[index], intersection, rng,
                    paths.color_buffer[index], mat);
}

__global__ void final_gathering_kernel(PathsView paths, glm::vec3* color_buffer,
                                       Normal* normal_buffer,
                                       float* depth_buffer,
                                       std::size_t iteration)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= paths.paths_count) return;

  const int pixel_index = paths.pixel_indices[index];

  final_gather(iteration, paths.color_buffer[index], paths.normal_buffer[index],
               paths.depth_buffer[index], color_buffer[pixel_index],
               normal_buffer[pixel_index], depth_buffer[pixel_index]);
}

enum class BufferNormalizationMethod { none, neg1_1_to_0_1 };

__global__ void preview_depth_kernel(UResolution resolution,
                                     const float* depth_buffer, uchar4* pbo)
{
  const auto [width, height] = resolution;
  const auto [x, y] = cuda::calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = cuda::flattern_index({x, y}, width, height);

  const float depth = depth_buffer[index];
  glm::vec3 color{1 / depth};

  constexpr auto color_float_to_255 = [](float v) {
    return static_cast<unsigned char>(glm::clamp(v, 0.f, 1.f) * 255.99f);
  };

  color = linear_to_gamma(color);

  if (x <= width && y <= height) {
    pbo[index] =
        uchar4{color_float_to_255(color.x), color_float_to_255(color.y),
               color_float_to_255(color.z), 1};
  }
}

__global__ void preview_kernel(UResolution resolution,
                               BufferNormalizationMethod normalization_method,
                               const glm::vec3* buffer, uchar4* pbo)
{
  const auto [width, height] = resolution;
  const auto [x, y] = cuda::calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = cuda::flattern_index({x, y}, width, height);

  auto color = buffer[index];

  switch (normalization_method) {
  case BufferNormalizationMethod::neg1_1_to_0_1: color = color * 0.5f + 0.5f;
  default: break;
  }

  constexpr auto color_float_to_255 = [](float v) {
    return static_cast<unsigned char>(glm::clamp(v, 0.f, 1.f) * 255.99f);
  };

  color = linear_to_gamma(color);

  if (x <= width && y <= height) {
    pbo[index] =
        uchar4{color_float_to_255(color.x), color_float_to_255(color.y),
               color_float_to_255(color.z), 255};
  }
}

PathTracer::PathTracer() = default;

void PathTracer::path_trace(const Camera& camera, UResolution resolution)
{
  if (iteration_ < max_iterations) {
    const auto [width, height] = resolution;
    const unsigned int pixels_count = width * height;

    const AggregateView aggregate_view{dev_scene_.aggregate};

    if (current_gpu_method == GPUMethod::megakernel) {
      const auto [width, height] = resolution;

      const dim3 block_size(8, 8);
      const dim3 blocks_per_grid( //
          (width + block_size.x - 1) / block_size.x,
          (height + block_size.y - 1) / block_size.y);

      const auto gpu_camera = camera.to_gpu_camera(resolution);
      cudaMemcpyToSymbol(constant_memory::gpu_camera, &gpu_camera,
                         sizeof(GPUCamera));

      path_tracing_mega_kernel<<<blocks_per_grid, block_size>>>(
          iteration_, aggregate_view, dev_scene_.materials.data(),
          dev_color_buffer_.data(), dev_normal_buffer_.data(),
          dev_depth_buffer_.data());
      cuda::check_CUDA_error("Path Tracing mega kernel");
    } else { // wavefront
      generate_rays(iteration_, camera, resolution,
                    PathsView{paths_, pixels_count});

      unsigned int paths_count = pixels_count;
      for (int i = 0; i < max_bounces && paths_count > 0; ++i) {
        //        fmt::print("Path count for iteration {} bounce {}: {}\n",
        //        iteration_, i,
        //                   paths_count);

        const unsigned int block_size = 64;
        const unsigned int block_count =
            (paths_count + block_size - 1) / block_size;

        const PathsView paths_view{paths_, paths_count};
        intersection_kernel<<<block_count, block_size>>>(
            paths_view, dev_intersection_buffer_.data(), aggregate_view);
        cuda::check_CUDA_error(
            fmt::format("Path Tracing intersection kernel bounce {}", i + 1));

        const auto paths_begin = thrust::make_zip_iterator(
            paths_view.rays, paths_view.pixel_indices, paths_view.color_buffer,
            paths_view.normal_buffer, paths_view.depth_buffer,
            paths_view.bounces_left_buffer);
        auto paths_end = paths_begin + paths_count;

        //        thrust::sort_by_key(
        //            thrust::device, dev_intersection_buffer_.data(),
        //            dev_intersection_buffer_.data() + paths_count,
        //            paths_begin,
        //            [] __device__(const Intersection& lhs, const Intersection&
        //            rhs) {
        //              return lhs.material_id < rhs.material_id;
        //            });

        material_kernel<<<block_count, block_size>>>(
            iteration_, i, paths_view, dev_scene_.materials.data(),
            dev_intersection_buffer_.data());
        cuda::check_CUDA_error(fmt::format("Material kernel bounce {}", i + 1));

        // Partition out terminated rays
        paths_end = thrust::stable_partition(
            thrust::device, paths_begin, paths_end,
            [] __device__(auto elem) { return thrust::get<5>(elem) > 0; });
        paths_count = paths_end - paths_begin;
      }

      {
        const unsigned int paths_count = pixels_count;
        const unsigned int block_size = 64;
        const unsigned int block_count =
            (paths_count + block_size - 1) / block_size;

        final_gathering_kernel<<<block_count, block_size>>>(
            PathsView{paths_, paths_count}, dev_color_buffer_.data(),
            dev_normal_buffer_.data(), dev_depth_buffer_.data(), iteration_);
        cuda::check_CUDA_error("Final Gathering kernel");
      }
    }

    ++iteration_;
  }

  path_trace_result_buffer_ = dev_color_buffer_.data();
}

void PathTracer::denoise(UResolution resolution)
{
  path_trace_result_buffer_ = atrous_denoiser.denoise(
      resolution.width, resolution.height, dev_color_buffer_.data(),
      dev_normal_buffer_.data(), dev_depth_buffer_.data(),
      dev_denoised_buffer_.data(), dev_denoised_buffer2_.data());
}

void PathTracer::send_to_preview(uchar4* dev_pbo, UResolution resolution,
                                 DisplayBufferType display_type) const
{
  constexpr unsigned int block_size = 16;

  const dim3 threads_per_block(block_size, block_size);
  const auto blocks_x = (resolution.width + block_size - 1) / block_size;
  const auto blocks_y = (resolution.height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  switch (display_type) {
  case DisplayBufferType::final: {
    preview_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        resolution, BufferNormalizationMethod::none, path_trace_result_buffer_,
        dev_pbo);
  } break;
  case DisplayBufferType::color:
    preview_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        resolution, BufferNormalizationMethod::none, dev_color_buffer_.data(),
        dev_pbo);
    break;
  case DisplayBufferType::normal:
    preview_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        resolution, BufferNormalizationMethod::neg1_1_to_0_1,
        dev_normal_buffer_.data(), dev_pbo);
    break;
  case DisplayBufferType::depth:
    preview_depth_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        resolution, dev_depth_buffer_.data(), dev_pbo);
    break;
  }
  cuda::check_CUDA_error("Preview kernel");
  CUDA_CHECK(cudaDeviceSynchronize());
}

void PathTracer::restart()
{
  iteration_ = 0;
}

void PathTracer::resize_image(UResolution resolution)
{
  paths_.resize_image(resolution);

  const auto [width, height] = resolution;
  const auto image_size = width * height;
  dev_color_buffer_ = cuda::make_buffer<glm::vec3>(image_size);
  dev_normal_buffer_ = cuda::make_buffer<Normal>(image_size);
  dev_depth_buffer_ = cuda::make_buffer<float>(image_size);
  dev_intersection_buffer_ = cuda::make_buffer<Intersection>(image_size);
  dev_denoised_buffer_ = cuda::make_buffer<glm::vec3>(image_size);
  dev_denoised_buffer2_ = cuda::make_buffer<glm::vec3>(image_size);

  CUDA_CHECK(cudaDeviceSynchronize());
  restart();
}

void Paths::resize_image(UResolution resolution)
{
  const auto [width, height] = resolution;
  const auto image_size = width * height;
  rays = cuda::make_buffer<Ray>(image_size);
  pixel_indices = cuda::make_buffer<int>(image_size);
  color_buffer = cuda::make_buffer<glm::vec3>(image_size);
  normal_buffer = cuda::make_buffer<Normal>(image_size);
  depth_buffer = cuda::make_buffer<float>(image_size);
  bounces_left_buffer = cuda::make_buffer<std::uint8_t>(image_size);
}

void PathTracer::create_buffers(UResolution resolution,
                                const SceneDescription& scene_description)
{
  dev_scene_ = scene_description.build_scene();
  resize_image(resolution);
}
