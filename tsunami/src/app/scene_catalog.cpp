// Purpose: Implements scene alias mapping and validation of user-provided scene arguments.
#include "app/internal/scene_catalog.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct SceneAlias {
	std::string_view key;
	std::string_view relative_path;
};

constexpr std::array<SceneAlias, 8> kSceneAliases = {
    SceneAlias{"", "resources/scenes/cornell/cornell.glb"},
    SceneAlias{"pool", "resources/scenes/poolHouse/poolHouse_optimized.glb"},
    SceneAlias{"pool_and_water", "resources/scenes/inflatable_pool/pool_and_water.glb"},
    SceneAlias{"poolandwater", "resources/scenes/inflatable_pool/pool_and_water.glb"},
    SceneAlias{"inflatable_pool", "resources/scenes/inflatable_pool/pool_and_water.glb"},
    SceneAlias{"inflatablepool", "resources/scenes/inflatable_pool/pool_and_water.glb"},
    SceneAlias{"chess", "resources/scenes/ABeautifulGame/glTF-Binary/ABeautifulGame.glb"},
    SceneAlias{"cornell", "resources/scenes/cornell/cornell.glb"},
};

constexpr std::array<SceneAlias, 2> kAdditionalAliases = {
    SceneAlias{"cornellsimple", "resources/scenes/cornell/cornell_simple.glb"},
    SceneAlias{"sponza", "resources/scenes/Sponza/glTF/Sponza.gltf"},
};

constexpr const char* kSceneUsage =
    "[pool|pool_and_water|chess|cornell|cornellsimple|sponza|<path/to/scene.gltf|.glb>]";

std::string toLowerAscii(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
		return static_cast<char>(std::tolower(character));
	});
	return value;
}

std::string resolveAliasPath(const std::string& scene_key) {
	for (const SceneAlias& alias : kSceneAliases) {
		if (scene_key == alias.key) {
			return std::string(alias.relative_path);
		}
	}
	for (const SceneAlias& alias : kAdditionalAliases) {
		if (scene_key == alias.key) {
			return std::string(alias.relative_path);
		}
	}
	return {};
}

}        // namespace

namespace app::scene {

const char* sceneArgumentUsage() {
	return kSceneUsage;
}

std::string resolveScenePathOrThrow(const std::string& scene_argument) {
	namespace fs = std::filesystem;

	const std::string scene_key    = toLowerAscii(scene_argument);
	const std::string aliased_path = resolveAliasPath(scene_key);
	if (!aliased_path.empty()) {
		return aliased_path;
	}

	const fs::path    user_path(scene_argument);
	const std::string extension = toLowerAscii(user_path.extension().string());
	if (extension != ".gltf" && extension != ".glb") {
		throw std::runtime_error("unknown scene alias '" + scene_argument +
		                         "'. Use one of: pool, pool_and_water, chess, cornell, "
		                         "cornellsimple, sponza, or provide a .gltf/.glb file path.");
	}

	std::error_code error;
	if (!fs::exists(user_path, error) || error) {
		throw std::runtime_error("scene file does not exist: " + user_path.string());
	}
	if (!fs::is_regular_file(user_path, error) || error) {
		throw std::runtime_error("scene path is not a regular file: " + user_path.string());
	}

	return user_path.lexically_normal().string();
}

}        // namespace app::scene
