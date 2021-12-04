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

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtx/compatibility.hpp>

#include "intersections.cuh"

static constexpr Object objects[] = {
    {ObjectType::sphere, 0}, {ObjectType::sphere, 1},   {ObjectType::sphere, 2},
    {ObjectType::sphere, 3}, {ObjectType::triangle, 0}, {ObjectType::mesh, 0},
};
static constexpr std::uint32_t material_indices[] = {0, 1, 2, 3, 1, 1};

static const Sphere spheres[] = {
    {{0.0f, -100.5f, -1.0f}, 100.f},
    {{0.0f, 0.0f, -1.0f}, 0.5f},
    {{-1.0f, 0.0f, -1.0f}, 0.5f},
    {{1.0f, 0.0f, -1.0f}, 0.5f},
};

static const Triangle triangles[] = {{
    {0.0f, 0.0f, 2.0f},
    {0.0f, 10.0f, 2.0f},
    {10.0f, 0.0f, 2.0f},
}};

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

[[nodiscard]] __device__ constexpr auto calculate_index_2d() -> Index2D
{
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  return Index2D{x, y};
}

[[nodiscard]] __device__ auto raygen(glm::mat4 camera_matrix, float fov,
                                     unsigned int width, unsigned int height,
                                     unsigned int x, unsigned int y,
                                     thrust::default_random_engine& rng) -> Ray
{
  const float aspect_ratio =
      static_cast<float>(width) / static_cast<float>(height);

  const float viewport_height = 2.0f * tan(fov / 2);
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
  return Ray{world_origin, 1e-4, world_direction, FLT_MAX};
}

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

__device__ auto ray_object_intersection_test(Ray ray, Object obj,
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
    const Object obj = objects[i];
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
                                  glm::vec3& color, Span<const Material> mat,
                                  Span<const DiffuseMateral> diffuse_mat,
                                  Span<const MetalMaterial> metal_mat,
                                  Span<const DielectricMaterial> dielectric_mat)
{
  ray.origin = record.point - 1e-4f *
                                  glm::sign(dot(ray.direction, record.normal)) *
                                  record.normal;
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

__global__ void path_tracing_kernel(
    unsigned int width, unsigned int height, glm::mat4 camera_matrix, float fov,
    glm::vec3* image, std::size_t iteration, AggregateView aggregate,
    Span<const Material> mat, Span<const DiffuseMateral> diffuse_mat,
    Span<const MetalMaterial> metal_mat,
    Span<const DielectricMaterial> dielectric_mat, const Vertex* vertices,
    Span<const std::uint32_t> indices)
{
  const auto [x, y] = calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = x + ((height - y) * width);

  thrust::default_random_engine rng(hash(hash(index) ^ iteration));

  auto ray = raygen(camera_matrix, fov, width, height, x, y, rng);

  // Path tracing
  glm::vec3 color{1.0f, 1.0f, 1.0f};
  for (int i = 0; i < 50; ++i) {
    HitRecord record;
    const bool hit =
        ray_scene_intersection_test(ray, aggregate, vertices, indices, record);
    if (!hit) {
      color *= get_background_color(ray);
      break;
    }
    evaluate_material(ray, record, rng, color, mat, diffuse_mat, metal_mat,
                      dielectric_mat);
  }

  color = gamma_correction(color);

  // Final gathering
  const auto sample_count = static_cast<float>(iteration + 1);
  image[index] = (image[index] * (sample_count - 1) + color) / sample_count;
}

__global__ void preview_kernel(unsigned int width, unsigned int height,
                               glm::vec3* image, uchar4* pbo)
{
  const auto [x, y] = calculate_index_2d();
  if (x >= width || y >= height) return;
  const auto index = x + ((height - y) * width);

  constexpr auto color_float_to_255 = [](float v) {
    return static_cast<unsigned char>(glm::clamp(v, 0.f, 1.f) * 255.99f);
  };

  const auto color = image[index];
  if (x <= width && y <= height) {
    pbo[index] =
        uchar4{color_float_to_255(color.x), color_float_to_255(color.y),
               color_float_to_255(color.z), 1};
  }
}

[[nodiscard]] static auto load_obj(const char* filename) -> Mesh
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

  Mesh mesh_gpu;
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
      width, height, camera.camera_matrix(), camera.fov(), dev_image_.data(),
      iteration_, AggregateView{aggregate_},
      Span{dev_mat_.data(), std::size(mat)},
      Span{dev_diffuse_mat_.data(), std::size(diffuse_mat)},
      Span{dev_metal_mat_.data(), std::size(metal_mat)},
      Span{dev_dielectric_mat_.data(), std::size(dielectric_mat)},
      cube_.vertices.data(), Span{cube_.indices.data(), cube_.indices_count});
  check_CUDA_error("Path Tracing kernel");
  preview_kernel<<<full_blocks_per_grid, threads_per_block>>>(
      width, height, dev_image_.data(), dev_pbo);
  check_CUDA_error("Preview kernel");

  CUDA_CHECK(cudaDeviceSynchronize());

  ++iteration_;
}

void PathTracer::restart()
{
  iteration_ = 0;
}

void PathTracer::resize_image(unsigned int width, unsigned int height)
{
  dev_image_ = cuda::make_buffer<glm::vec3>(width * height);
  CUDA_CHECK(cudaDeviceSynchronize());
  restart();
}

template <typename T>
[[nodiscard]] static auto create_buffer_from_cpu_data(Span<const T> span)
{
  auto dev_buffer = cuda::make_buffer<T>(span.size());
  CUDA_CHECK(cudaMemcpy(dev_buffer.data(), span.data(), span.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return dev_buffer;
}

void PathTracer::create_buffers(unsigned int width, unsigned int height)
{
  aggregate_.objects = create_buffer_from_cpu_data(Span{objects});
  aggregate_.object_count = std::size(objects);
  aggregate_.object_material_indices =
      create_buffer_from_cpu_data(Span{material_indices});

  aggregate_.spheres = create_buffer_from_cpu_data(Span{spheres});
  aggregate_.sphere_count = std::size(spheres);

  aggregate_.triangles = create_buffer_from_cpu_data(Span{triangles});
  aggregate_.triangle_count = std::size(triangles);

  dev_mat_ = create_buffer_from_cpu_data(Span{mat});
  dev_diffuse_mat_ = create_buffer_from_cpu_data(Span{diffuse_mat});
  dev_metal_mat_ = create_buffer_from_cpu_data(Span{metal_mat});
  dev_dielectric_mat_ = create_buffer_from_cpu_data(Span{dielectric_mat});
  resize_image(width, height);
}
