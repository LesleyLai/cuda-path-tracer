#include "options.hpp"

#include <fmt/format.h>

#include <cxxopts.hpp>

[[nodiscard]] auto parse_cli_args(int argc, char** argv) -> Options
try {
  cxxopts::Options options("cuda_pt", "A Path Tracer written in CUDA");

  // clang-format off
  options.add_options()
  ("filename", "The name of the scene file",cxxopts::value<std::string>())
  ("i,interactive", "Open cuda_pt as an interactive app")
  ("h,help", "Print this message");
  // clang-format on

  options.positional_help("<filename>");
  options.parse_positional({"filename"});

  const auto result = options.parse(argc, argv);
  if (result.count("help")) {
    fmt::print("{}\n", options.help());
    exit(0);
  }

  return Options{
      .is_interactive = result["interactive"].as<bool>(),
      .filename = result["filename"].as<std::string>(),
  };
} catch (const cxxopts::OptionException& e) {
  fmt::print("{}\n", e.what());
  std::exit(1);
}