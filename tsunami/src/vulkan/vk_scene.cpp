#include "tsunami/vulkan/vk_scene.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>

static std::string toLowerAscii(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

std::string resolveScenePathOrThrow(const std::string& scene_argument) {
	namespace fs = std::filesystem;

	if (scene_argument.empty())
		return "resources/scenes/cornell/cornell.glb";

	const std::string scene_key = toLowerAscii(scene_argument);
	if (scene_key == "pool")
		return "resources/scenes/poolHouse/poolHouse_optimized.glb";
	if (scene_key == "chess")
		return "resources/scenes/ABeautifulGame/glTF-Binary/ABeautifulGame.glb";
	if (scene_key == "cornell")
		return "resources/scenes/cornell/cornell.glb";
	if (scene_key == "cornellsimple")
		return "resources/scenes/cornell/cornell_simple.glb";
	if (scene_key == "sponza")
		return "resources/scenes/Sponza/glTF/Sponza.gltf";

	const fs::path    user_path(scene_argument);
	const std::string ext = toLowerAscii(user_path.extension().string());
	if (ext != ".gltf" && ext != ".glb") {
		throw std::runtime_error(
		    "unknown scene alias '" + scene_argument +
		    "'. Use one of: pool, chess, cornell, cornellsimple, sponza, or provide a .gltf/.glb "
		    "file path.");
	}

	std::error_code ec;
	if (!fs::exists(user_path, ec) || ec)
		throw std::runtime_error("scene file does not exist: " + user_path.string());
	if (!fs::is_regular_file(user_path, ec) || ec)
		throw std::runtime_error("scene path is not a regular file: " + user_path.string());

	return user_path.lexically_normal().string();
}
