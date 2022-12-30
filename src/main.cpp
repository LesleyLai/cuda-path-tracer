#include "cli/cli.hpp"
#include "interactive-app/app.hpp"
#include "lib/assets/assets.hpp"
#include "lib/assets/scene_parser.hpp"
#include "lib/prelude.hpp"

#include <fmt/format.h>

auto main(int argc, char** argv) -> int
try {
  const CliConfigurations cli_configs = parse_cli_args(argc, argv);
  const auto asset_path = locate_asset_path(std::filesystem::current_path());

  if (cli_configs.is_interactive) {
    const SceneDescription scene_desc = read_scene(cli_configs, asset_path);
    App app{scene_desc};
    app.main_loop();
  } else {
    execute_cli_version(cli_configs, asset_path);
  }

} catch (const std::exception& e) {
  fmt::print(stderr, "cuda_pt fatal error: Unhandled exception: {}\n",
             e.what());
}
