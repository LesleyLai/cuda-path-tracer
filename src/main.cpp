#include "interactive-app/app.hpp"
#include "lib/options.hpp"

#include "lib/scene_parser.hpp"

#include <cstdio>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

void run(const Options& options)
{
  int gpu_device = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpu_device > device_count) {
    fmt::print(stderr, "Error: GPU device number is greater than the number of "
                       "devices! Perhaps a CUDA-capable GPU is not installed?");
    std::exit(1);
  }

  const unsigned int width = 800, height = 800;
  const UResolution resolution{.width = width, .height = height};

  fmt::print("file {}\n", options.filename);

  SceneDescription scene_desc = read_scene(options.filename);
  Camera camera;
  camera.set_position(glm::vec3(10, 10, 0));

  PathTracer path_tracer{options};
  path_tracer.create_buffers(resolution, scene_desc);
  path_tracer.path_trace(camera, resolution);

  uchar4* buffer = nullptr;
  CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&buffer),
                               width * height * sizeof(uchar4)));
  path_tracer.send_to_preview(buffer, resolution);

  CUDA_CHECK(cudaDeviceSynchronize());

  if (stbi_write_png("output.png", width, height, 4, buffer, 0) == 0) {
    fmt::print(stderr, "Filed to write to file output.png");
  }

  cudaFree(buffer);

  fmt::print("Done path tracing {}!\n", options.filename);
}

auto main(int argc, char** argv) -> int
try {
  const Options options = parse_cli_args(argc, argv);

  if (options.is_interactive) {
    App app{options};
    app.main_loop();
  } else {
    run(options);
  }

} catch (const std::exception& e) {
  fmt::print(stderr, "cuda_pt fatal error: Unhandled exception: {}\n",
             e.what());
}
