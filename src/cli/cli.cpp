#include "cli.hpp"

#include <fmt/format.h>
#include <stb_image_write.h>

#include "lib/camera.hpp"
#include "lib/path_tracer.hpp"
#include "lib/scene_parser.hpp"

void execute_cli_version(const Options& options)
{
  int gpu_device = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpu_device > device_count) {
    fmt::print(stderr, "Error: GPU device number is greater than the number of "
                       "devices! Perhaps a CUDA-capable GPU is not installed?");
    std::exit(1);
  }

  const SceneDescription scene_desc = read_scene(options);
  const UResolution resolution = scene_desc.resolution.to_unsigned();
  const auto [width, height] = resolution;

  const int spp = scene_desc.spp;

  const Camera camera{
      .position = glm::vec3(0, 0, 0),
      .fov = glm::pi<float>() / 2,
  };

  PathTracer path_tracer{};
  path_tracer.create_buffers(resolution, scene_desc);
  CUDA_CHECK(cudaDeviceSynchronize());

  fmt::print("Start path tracing\n");
  fmt::print("spp: {}\n", spp);
  fmt::print("width: {}, height: {}\n", width, height);
  std::fflush(stdout);

  for (int i = 0; i < spp; ++i) {
    path_tracer.path_trace(camera, resolution);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  auto buffer = cuda::make_managed_buffer<uchar4>(width * height);
  path_tracer.send_to_preview(buffer.data(), resolution);

  CUDA_CHECK(cudaDeviceSynchronize());

  if (stbi_write_png("output.png", width, height, 4, buffer.data(), 0) == 0) {
    fmt::print(stderr, "Filed to write to file output.png");
  }

  fmt::print("Done path tracing {}!\n", options.filename);
}