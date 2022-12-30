#include "configurations.hpp"

#include <fmt/format.h>

#include <cxxopts.hpp>

[[nodiscard]] auto parse_cli_args(int argc, char** argv) -> CliConfigurations
try {
  cxxopts::Options options("cuda_pt", "A Path Tracer written in CUDA");

  std::optional<std::string> output_file;
  std::optional<int> spp;

  // clang-format off
  options.add_options()
  ("filename", "The name of the scene file", cxxopts::value<std::string>())
  ("o,output", "Output path tracing result to a file", cxxopts::value<std::optional<std::string>>(output_file))
  ("h,help", "Print this message")
  ("spp",
    "Sample per pixel (if provided, this value will overwrite the setting in the scene file",
    cxxopts::value<std::optional<int>>(spp));
  // clang-format on

  options.positional_help("<filename>");
  options.parse_positional({"filename"});

  const auto result = options.parse(argc, argv);
  if (result.count("help")) {
    fmt::print("{}\n", options.help());
    std::exit(0);
  }

  if (not result.count("filename")) {
    fmt::print(stderr, "Usage: cuda_pt [options] <filename>\n");
    fmt::print(stderr, "Run 'cuda_pt --help' for more information");
    std::exit(1);
  }

  return CliConfigurations{.filename = result["filename"].as<std::string>(),
                           .spp = spp,
                           .output_filename = output_file};
} catch (const cxxopts::OptionException& e) {
  fmt::print(stderr, "{}\n", e.what());
  std::exit(1);
}