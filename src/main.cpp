#include "cli/cli.hpp"
#include "interactive-app/app.hpp"
#include "lib/assets.hpp"
#include "lib/prelude.hpp"
#include "lib/scene_parser.hpp"

#include <cstdio>

auto main(int argc, char** argv) -> int
try {
  const CliConfigurations cli_configs = parse_cli_args(argc, argv);
  const auto asset_path = locate_asset_path(std::filesystem::current_path());
  const SceneDescription scene_desc = read_scene(cli_configs, asset_path);

  if (cli_configs.is_interactive) {
    App app{scene_desc};
    app.main_loop();
  } else {
    execute_cli_version(scene_desc);
  }

} catch (const std::exception& e) {
  fmt::print(stderr, "cuda_pt fatal error: Unhandled exception: {}\n",
             e.what());
}
