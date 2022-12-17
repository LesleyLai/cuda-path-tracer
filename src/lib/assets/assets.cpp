#include "assets.hpp"
#include "lib/prelude.hpp"

#include <optional>

auto locate_asset_path(const std::filesystem::path& current_path)
    -> std::filesystem::path
{
  namespace fs = std::filesystem;

  std::optional<fs::path> result = std::nullopt;
  for (auto path = current_path; path != current_path.root_path();
       path = path.parent_path()) {
    const auto assets_path = path / "assets";
    if (exists(assets_path) && is_directory(assets_path)) {
      result = assets_path;
    }
  }

  if (!result.has_value()) { panic("Cannot find assets directory"); }

  return absolute(*result);
}