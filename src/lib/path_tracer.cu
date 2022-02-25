#include "path_tracer.hpp"

#include "camera.hpp"
#include "constant_memory.cuh"
#include "cuda_utils/2d_indices.cuh"
#include "cuda_utils/cuda_buffer.hpp"
#include "distributions.cuh"
#include "ray_gen.cuh"
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

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtx/compatibility.hpp>

#include "intersections.cuh"

__device__ auto get_background_color(Ray r) -> glm::vec3
{
  const glm::vec3 unit_direction = glm::normalize(r.direction);
  const auto t = 0.5f * (unit_direction.y + 1.0f);
  return glm::lerp(glm::vec3(0.5, 0.7, 1.0), glm::vec3(1.0, 1.0, 1.0), t);
}

__device__ auto ray_mesh_intersection_test(Ray ray, const Vertex* vertices,
                                           Span<const std::uint32_t> indices,
                                           HitRecord& record) -> bool
{
  bool hit = false;
  for (std::size_t j = 0; j < indices.size(); j += 3) {
    const auto index0 = indices[j];
    const auto index1 = indices[j + 1];
    const auto index2 = indices[j + 2];

    const auto p0 = vertices[index0].position;
    const auto p1 = vertices[index1].position;
    const auto p2 = vertices[index2].position;

    if (ray_triangle_intersection_test(ray, p0, p1, p2, record)) {
      hit = true;
      ray.t_max = record.t;
    }
  }
  return hit;
}

__device__ auto ray_object_intersection_test(Ray ray, GPUObject obj,
                                             AggregateView aggregate,
                                             const Vertex* vertices,
                                             Span<const std::uint32_t> indices,
                                             HitRecord& record) -> bool
{
  switch (obj.type) {
  case ObjectType::sphere: {
    const auto sphere = aggregate.spheres[obj.index];
    return ray_sphere_intersection_test(ray, sphere, record);
  }
  case ObjectType::triangle: {
    const auto triangle = aggregate.triangles[obj.index];
    return ray_triangle_intersection_test(ray, triangle.pt0, triangle.pt1,
                                          triangle.pt2, record);
  }
  case ObjectType::mesh:
    return ray_mesh_intersection_test(ray, vertices, indices, record);
  }
  // unreachable
  return false;
}

__device__ auto ray_scene_intersection_test(Ray ray, AggregateView aggregate,
                                            const Vertex* vertices,
                                            Span<const std::uint32_t> indices,
                                            HitRecord& record) -> bool
{
  bool hit = false;

  const auto objects = aggregate.objects;
  const auto* object_material_indices = aggregate.object_material_indices;

  for (std::size_t i = 0; i < objects.size(); ++i) {
    const GPUObject obj = objects[i];
    if (ray_object_intersection_test(ray, obj, aggregate, vertices, indices,
                                     record)) {
      hit = true;
      record.material_id = object_material_indices[i];
      ray.t_max = record.t;
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

__device__ void evaluate_material(Ray& ray, const HitRecord record,
                                  thrust::default_random_engine& rng,
                                  glm::vec3& color, const Material* materials)
{
  ray.origin = record.point - 1e-4f *
                                  glm::sign(dot(ray.direction, record.normal)) *
                                  record.normal;
  // material stuff
  const Material& material = materials[record.material_id];
  switch (material.type) {
  case Material::Type::Diffuse: {
    auto diffuse = material.data.diffuse;
    auto scatter_direction =
        glm::normalize(record.normal + random_in_unit_sphere(rng));

    // Catch degenerated case
    if (abs(scatter_direction.x) < 1e-8 && abs(scatter_direction.y) < 1e-8 &&
        abs(scatter_direction.z) < 1e-8) {
      scatter_direction = record.normal;
    }

    ray.direction = scatter_direction;
    color *= diffuse.albedo;
  } break;
  case Material::Type::Metal: {
    const auto metal = material.data.metal;
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
    const auto dielectric = material.data.dielectric;
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

    ray = Ray{record.point, 1e-5, direction, std::numeric_limits<float>::max()};
  } break;
  }
}

[[nodiscard]] __device__ auto gamma_correction(glm::vec3 color) -> glm::vec3
{
  color.x = glm::pow(color.x, 1.f / 2.2f);
  color.y = glm::pow(color.y, 1.f / 2.2f);
  color.z = glm::pow(color.z, 1.f / 2.2f);
  return color;
}

__global__ void path_tracing_kernel(glm::vec3* color_buffer,
                                    glm::vec3* normal_buffer,
                                    float* depth_buffer, std::size_t iteration,
                                    AggregateView aggregate,
                                    const Material* mat, const Vertex* vertices,
                                    Span<const std::uint32_t> indices)
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

  // Path tracing
  glm::vec3 color{1.0f, 1.0f, 1.0f};
  glm::vec3 normal = -ray.direction;
  float depth = 1e6;

  for (int i = 0; i < 50; ++i) {
    HitRecord record;
    const bool hit =
        ray_scene_intersection_test(ray, aggregate, vertices, indices, record);
    if (!hit) {
      color *= get_background_color(ray);
      break;
    }

    evaluate_material(ray, record, rng, color, mat);
    if (i == 0) {
      normal = record.normal;
      depth = record.t;
    }
  }

  // Final gathering
  const auto sample_count = static_cast<float>(iteration + 1);
  const auto temporal_accumulate = [sample_count](auto value, auto current) {
    return (value * (sample_count - 1) + current) / sample_count;
  };

  if (iteration == 0) {
    color_buffer[index] = color;
    normal_buffer[index] = normal;
    depth_buffer[index] = depth;
  } else {
    color_buffer[index] = temporal_accumulate(color_buffer[index], color);
    normal_buffer[index] = temporal_accumulate(normal_buffer[index], normal);
    depth_buffer[index] = temporal_accumulate(depth_buffer[index], depth);
  }
}

enum class BufferNormalizationMethod { none, neg1_1_to_0_1 };

__global__ void preview_depth_kernel(unsigned int width, unsigned int height,
                                     const float* depth_buffer, uchar4* pbo)
{
  const auto [x, y] = cuda::calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = cuda::flattern_index({x, y}, width, height);

  const float depth = depth_buffer[index];
  glm::vec3 color{1 / depth};

  constexpr auto color_float_to_255 = [](float v) {
    return static_cast<unsigned char>(glm::clamp(v, 0.f, 1.f) * 255.99f);
  };

  color = gamma_correction(color);

  if (x <= width && y <= height) {
    pbo[index] =
        uchar4{color_float_to_255(color.x), color_float_to_255(color.y),
               color_float_to_255(color.z), 1};
  }
}

__global__ void preview_kernel(unsigned int width, unsigned int height,
                               BufferNormalizationMethod normalization_method,
                               const glm::vec3* buffer, uchar4* pbo)
{
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

  color = gamma_correction(color);

  if (x <= width && y <= height) {
    pbo[index] =
        uchar4{color_float_to_255(color.x), color_float_to_255(color.y),
               color_float_to_255(color.z), 1};
  }
}

[[nodiscard]] static auto load_obj(const char* filename) -> GPUMesh
{
  Assimp::Importer importer;

  const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate);
  if (!scene || !scene->HasMeshes()) {
    throw std::runtime_error(fmt::format("Unable to load {}", filename));
  }

  const aiMesh* mesh = scene->mMeshes[0];

  thrust::host_vector<Vertex> vertices;
  for (unsigned i = 0; i != mesh->mNumVertices; i++) {
    const aiVector3D v = mesh->mVertices[i];
    // const aiVector3D n = mesh->mNormals[i];
    // const aiVector3D t = mesh->mTextureCoords[0][i];
    vertices.push_back(Vertex{{v.x + 10.f, v.z, v.y}});
  }

  thrust::host_vector<std::uint32_t> indices;
  for (unsigned i = 0; i != mesh->mNumFaces; i++)
    for (unsigned j = 0; j != 3; j++)
      indices.push_back(mesh->mFaces[i].mIndices[j]);

  GPUMesh mesh_gpu;
  mesh_gpu.vertices = cuda::make_buffer<Vertex>(vertices.size());
  mesh_gpu.indices = cuda::make_buffer<std::uint32_t>(indices.size());
  mesh_gpu.indices_count = indices.size();

  thrust::copy(vertices.begin(), vertices.end(),
               thrust::device_pointer_cast(mesh_gpu.vertices.data()));
  thrust::copy(indices.begin(), indices.end(),
               thrust::device_pointer_cast(mesh_gpu.indices.data()));

  return mesh_gpu;
}

PathTracer::PathTracer()
{
  cube_ = load_obj("models/cube.obj");
}

void PathTracer::path_trace(const Camera& camera, unsigned int width,
                            unsigned int height)
{
  constexpr unsigned int block_size = 16;

  const dim3 threads_per_block(block_size, block_size);
  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  auto gpu_camera = camera.generate_gpu_camera();
  gpu_camera.width = width;
  gpu_camera.height = height;
  cudaMemcpyToSymbol(constant_memory::gpu_camera, &gpu_camera,
                     sizeof(GPUCamera));

  if (iteration_ < max_iterations) {
    path_tracing_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        dev_color_buffer_.data(), dev_normal_buffer_.data(),
        dev_depth_buffer_.data(), iteration_,
        AggregateView{dev_scene_.aggregate}, dev_scene_.materials.data(),
        cube_.vertices.data(), Span{cube_.indices.data(), cube_.indices_count});
    cuda::check_CUDA_error("Path Tracing kernel");

    ++iteration_;
  }

  path_trace_result_buffer_ = dev_color_buffer_.data();
  if (enable_denoising) {
    path_trace_result_buffer_ = atrous_denoiser.denoise(
        width, height, dev_color_buffer_.data(), dev_normal_buffer_.data(),
        dev_depth_buffer_.data(), dev_denoised_buffer_.data(),
        dev_denoised_buffer2_.data());
  }
}

void PathTracer::send_to_preview(uchar4* dev_pbo, unsigned int width,
                                 unsigned int height) const
{
  constexpr unsigned int block_size = 16;

  const dim3 threads_per_block(block_size, block_size);
  const auto blocks_x = (width + block_size - 1) / block_size;
  const auto blocks_y = (height + block_size - 1) / block_size;
  const dim3 full_blocks_per_grid(blocks_x, blocks_y);

  switch (display_buffer_) {
  case DisplayBuffer::path_tracing: {
    preview_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        width, height, BufferNormalizationMethod::none,
        path_trace_result_buffer_, dev_pbo);
  } break;
  case DisplayBuffer::color:
    preview_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        width, height, BufferNormalizationMethod::none,
        dev_color_buffer_.data(), dev_pbo);
    break;
  case DisplayBuffer::normal:
    preview_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        width, height, BufferNormalizationMethod::neg1_1_to_0_1,
        dev_normal_buffer_.data(), dev_pbo);
    break;
  case DisplayBuffer::depth:
    preview_depth_kernel<<<full_blocks_per_grid, threads_per_block>>>(
        width, height, dev_depth_buffer_.data(), dev_pbo);
    break;
  }
  cuda::check_CUDA_error("Preview kernel");
  CUDA_CHECK(cudaDeviceSynchronize());
}

void PathTracer::restart()
{
  iteration_ = 0;
}

void PathTracer::resize_image(unsigned int width, unsigned int height)
{
  const auto image_size = width * height;
  dev_color_buffer_ = cuda::make_buffer<glm::vec3>(image_size);
  dev_normal_buffer_ = cuda::make_buffer<glm::vec3>(image_size);
  dev_depth_buffer_ = cuda::make_buffer<float>(image_size);
  dev_denoised_buffer_ = cuda::make_buffer<glm::vec3>(image_size);
  dev_denoised_buffer2_ = cuda::make_buffer<glm::vec3>(image_size);
  CUDA_CHECK(cudaDeviceSynchronize());
  restart();
}

void PathTracer::create_buffers(unsigned int width, unsigned int height,
                                const SceneDescription& scene_description)
{
  dev_scene_ = scene_description.build();
  resize_image(width, height);
}