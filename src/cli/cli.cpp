#include "cli.hpp"

#include <chrono>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include "../lib/assets/scene_parser.hpp"
#include "../lib/camera.hpp"
#include "../lib/image.hpp"
#include "../lib/path_tracer.hpp"

[[nodiscard]] auto now() -> std::chrono::steady_clock::time_point
{
  return std::chrono::steady_clock::now();
}

template <class Rep, class Period>
[[nodiscard]] auto
to_seconds(const std::chrono::duration<Rep, Period>& duration)
{
  return std::chrono::duration<double>(duration);
}

class Stopwatch {
  struct Entry {
    std::string name;
    std::chrono::steady_clock::duration duration;
  };

  std::chrono::steady_clock::time_point start_ =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last_ =
      std::chrono::steady_clock::now();
  std::vector<Entry> entries_;

public:
  Stopwatch() = default;
  void end_stage(std::string name)
  {
    const auto now = std::chrono::steady_clock::now();
    entries_.push_back(Entry{std::move(name), now - last_});
    last_ = now;
  }

  [[nodiscard]] auto report() const -> std::string
  {
    std::string result;
    fmt::format_to(std::back_inserter(result), "Elapsed time\n===========\n");
    for (auto& [name, duration] : entries_) {
      fmt::format_to(std::back_inserter(result), "{}: {}\n", name,
                     to_seconds(duration));
    }
    fmt::format_to(std::back_inserter(result), "Total: {}\n",
                   to_seconds(last_ - start_));
    return result;
  }
};

void execute_cli_version(const CliConfigurations& configs,
                         const std::filesystem::path& asset_path)
{
  Stopwatch stopwatch;

  const SceneDescription scene_desc = read_scene(configs, asset_path);

  stopwatch.end_stage("Scene loading");

  const int gpu_device = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpu_device > device_count) {
    fmt::print(stderr, "Error: GPU device number is greater than the number of "
                       "devices! Perhaps a CUDA-capable GPU is not installed?");
    std::exit(1);
  }

  const UResolution resolution = scene_desc.resolution.to_unsigned();
  const auto [width, height] = resolution;

  const int spp = scene_desc.spp;
  const auto& camera = scene_desc.camera;

  PathTracer path_tracer{};
  path_tracer.create_buffers(resolution, scene_desc);
  CUDA_CHECK(cudaDeviceSynchronize());

  fmt::print("Start path tracing\n");
  fmt::print("spp: {}\n", spp);
  fmt::print("width: {}, height: {}\n", width, height);

  stopwatch.end_stage("Initialization");

  path_tracer.max_iterations = spp;
  for (int i = 0; i < spp; ++i) {
    path_tracer.path_trace(camera, resolution);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  stopwatch.end_stage("Path Tracing");

  auto buffer = cuda::make_managed_buffer<uchar4>(width * height);
  path_tracer.send_to_preview(buffer.data(), resolution);

  CUDA_CHECK(cudaDeviceSynchronize());

  write_image_file(configs.output_filename.value(), resolution.to_signed(),
                   buffer.data());

  stopwatch.end_stage("Write image file");

  fmt::print("Done path tracing {}!\n\n", scene_desc.filename);
  fmt::print("{}\n", stopwatch.report());
}