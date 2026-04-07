#pragma once

#include <assimp/material.h>
#include <assimp/scene.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class Texture {
  public:
	Texture()  = default;
	~Texture() = default;

	static std::shared_ptr<Texture> load_from_file(const std::string& path, bool srgb = true);
	static std::shared_ptr<Texture> load_from_assimp(const aiScene*     scene,
	                                                 const std::string& scene_source_path,
	                                                 const aiString& tex_path, bool srgb = true);

	bool valid() const {
		return !pixels.empty();
	}

	int         width    = 0;
	int         height   = 0;
	int         channels = 4;
	bool        is_srgb  = true;
	std::string source_path;

	std::vector<std::uint8_t> pixels;
};
