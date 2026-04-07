#include "tsunami/texture/texture.h"

#include <assimp/texture.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

static std::shared_ptr<Texture> make_texture_from_rgba8(const unsigned char* data, int width,
                                                        int height, int channels_in_file,
                                                        const std::string& source_path, bool srgb) {
	auto tex         = std::make_shared<Texture>();
	tex->width       = width;
	tex->height      = height;
	tex->channels    = 4;
	tex->is_srgb     = srgb;
	tex->source_path = source_path;

	const size_t size = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
	tex->pixels.assign(data, data + size);
	return tex;
}

std::shared_ptr<Texture> Texture::load_from_file(const std::string& path, bool srgb) {
	stbi_set_flip_vertically_on_load(false);

	int            w = 0, h = 0, c = 0;
	unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 4);
	if (!data) {
		std::cerr << "[Texture] Failed to load: " << path << "\n";
		return nullptr;
	}

	auto tex = make_texture_from_rgba8(data, w, h, c, path, srgb);
	stbi_image_free(data);
	return tex;
}

std::shared_ptr<Texture> Texture::load_from_assimp(const aiScene*     scene,
                                                   const std::string& scene_source_path,
                                                   const aiString& tex_path, bool srgb) {
	if (!scene)
		return nullptr;

	std::string path = tex_path.C_Str();
	if (path.empty())
		return nullptr;

	// Embedded texture: "*0", "*1", ...
	if (!path.empty() && path[0] == '*') {
		const int embedded_index = std::atoi(path.c_str() + 1);
		if (embedded_index < 0 || embedded_index >= static_cast<int>(scene->mNumTextures))
			return nullptr;

		const aiTexture* ai_tex = scene->mTextures[embedded_index];
		if (!ai_tex)
			return nullptr;

		// Compressed embedded texture
		if (ai_tex->mHeight == 0) {
			int                  w = 0, h = 0, c = 0;
			const unsigned char* raw = reinterpret_cast<const unsigned char*>(ai_tex->pcData);
			unsigned char*       data =
			    stbi_load_from_memory(raw, static_cast<int>(ai_tex->mWidth), &w, &h, &c, 4);

			if (!data) {
				std::cerr << "[Texture] Failed to decode embedded compressed texture: " << path
				          << "\n";
				return nullptr;
			}

			auto tex = make_texture_from_rgba8(
			    data, w, h, c, scene_source_path + "|" + path, srgb);
			stbi_image_free(data);
			return tex;
		}

		// Raw embedded RGBA texels
		auto tex         = std::make_shared<Texture>();
		tex->width       = static_cast<int>(ai_tex->mWidth);
		tex->height      = static_cast<int>(ai_tex->mHeight);
		tex->channels    = 4;
		tex->is_srgb     = srgb;
		tex->source_path = scene_source_path + "|" + path;
		tex->pixels.resize(static_cast<size_t>(tex->width) * static_cast<size_t>(tex->height) * 4);

		for (int y = 0; y < tex->height; ++y) {
			for (int x = 0; x < tex->width; ++x) {
				const aiTexel& src     = ai_tex->pcData[y * tex->width + x];
				const size_t   dst_i   = (static_cast<size_t>(y) * tex->width + x) * 4;
				tex->pixels[dst_i + 0] = src.r;
				tex->pixels[dst_i + 1] = src.g;
				tex->pixels[dst_i + 2] = src.b;
				tex->pixels[dst_i + 3] = src.a;
			}
		}

		return tex;
	}

	fs::path full_path = fs::path(scene_source_path).parent_path() / fs::path(path);
	full_path          = full_path.lexically_normal();
	return load_from_file(full_path.string(), srgb);
}
