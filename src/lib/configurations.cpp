#include "configurations.hpp"

#include <fmt/format.h>

#include <cxxopts.hpp>

[[nodiscard]] auto parse_cli_args(int argc, char** argv) -> CliConfigurations
try {
  cxxopts::Options options("cuda_pt", "A Path Tracer written in CUDA");

  // clang-format off
  options.add_options()
  ("filename", "The name of the scene file",cxxopts::value<std::string>())
  ("i,interactive", "Open cuda_pt as an interactive app")
  ("h,help", "Print this message")
  ("spp", "Sample per pixel (if provided, this value will overwrite the setting in the scene file", cxxopts::value<int>());
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

  const auto spp = result.count("spp") == 0
                       ? std::nullopt
                       : std::optional<int>{result["spp"].as<int>()};

  return CliConfigurations{
      .is_interactive = result["interactive"].as<bool>(),
      .filename = result["filename"].as<std::string>(),
      .spp = spp,
  };
} catch (const cxxopts::OptionException& e) {
  fmt::print(stderr, "{}\n", e.what());
  std::exit(1);
}