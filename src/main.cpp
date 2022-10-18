#include "cli/cli.hpp"
#include "interactive-app/app.hpp"
#include "lib/scene_parser.hpp"

#include <cstdio>

auto main(int argc, char** argv) -> int
try {
  const Options options = parse_cli_args(argc, argv);

  if (options.is_interactive) {
    App app{options};
    app.main_loop();
  } else {
    execute_cli_version(options);
  }

} catch (const std::exception& e) {
  fmt::print(stderr, "cuda_pt fatal error: Unhandled exception: {}\n",
             e.what());
}
