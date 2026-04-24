// Purpose: Internal declarations for scene aliases, usage text, and path resolution helpers.
#pragma once

#include <string>

namespace app::scene {

std::string resolveScenePathOrThrow(const std::string& scene_argument);

const char* sceneArgumentUsage();

}        // namespace app::scene
