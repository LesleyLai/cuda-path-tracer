#include <filesystem>

#include "image.hpp"

#include <stb_image_write.h>

void write_image_file(std::string filename, Resolution res, const void* data)
{
  auto [width, height] = res;

  std::filesystem::path output_path{filename};

  if (output_path.extension() == ".png") {
    if (stbi_write_png(filename.c_str(), width, height, 4, data, 0) == 0) {
      SPDLOG_CRITICAL("Failed to write to image file {}", filename);
    }
  } else {
    fmt::print(stderr, "{} has an unrecognized extension\n", filename);
  }
}