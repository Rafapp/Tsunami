#include "tsunami/vulkan/vk_resources.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

// OpenPBR LUT data (large C++ arrays — compiled once here).
#include <glm/glm.hpp>
#define OPENPBR_LANGUAGE_TARGET_CPP 1
#define OPENPBR_USE_TEXTURE_LUTS 0
#define OPENPBR_ENERGY_TABLES_USE_UINT16 0
#include "impl/data/openpbr_energy_arrays.h"
#include "impl/data/openpbr_ltc_array.h"
#include "openpbr.h"
#include "openpbr_data_constants.h"
#undef OPENPBR_LANGUAGE_TARGET_CPP
#undef OPENPBR_USE_TEXTURE_LUTS
#undef OPENPBR_ENERGY_TABLES_USE_UINT16

#include "tsunami/texture/texture.h"

#include "tsunami/vulkan/vk_helpers.h"

// ============================================================
// LUT format conversion helpers
// ============================================================
static std::vector<float> uint_lut_to_rgba32f(const uint32_t* data, size_t count) {
	std::vector<float> out(count * 4, 0.f);
	for (size_t i = 0; i < count; ++i)
		out[i * 4] = float(data[i]) / 65535.f;
	return out;
}

static std::vector<float> float3_lut_to_rgba32f(const float* data, size_t count_vec3) {
	std::vector<float> out(count_vec3 * 4, 0.f);
	for (size_t i = 0; i < count_vec3; ++i) {
		out[i * 4 + 0] = data[i * 3 + 0];
		out[i * 4 + 1] = data[i * 3 + 1];
		out[i * 4 + 2] = data[i * 3 + 2];
	}
	return out;
}

// ============================================================
// Generic RGBA32F image upload helpers
// ============================================================
static void upload_rgba32f_2d(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const float* rgba, uint32_t w, uint32_t h, LutTexture2D& out) {
	VkDeviceSize  sz = (VkDeviceSize) w * h * 4 * sizeof(float);
	VmaAllocation sa;
	VkBuffer stg = create_and_upload_buffer(alloc, sz, rgba, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);

	VkImageCreateInfo ii{};
	ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ii.imageType     = VK_IMAGE_TYPE_2D;
	ii.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent        = {w, h, 1};
	ii.mipLevels     = 1;
	ii.arrayLayers   = 1;
	ii.samples       = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
	ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(alloc, &ii, &ai, &out.image, &out.alloc, nullptr);

	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkBufferImageCopy r{};
	r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	r.imageExtent      = {w, h, 1};
	vkCmdCopyBufferToImage(cmd, stg, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	end_one_time_cmd(dev, pool, q, cmd);

	out.view =
	    create_image_view(dev, out.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D);
	out.width  = w;
	out.height = h;
	vmaDestroyBuffer(alloc, stg, sa);
}

static void upload_rgba32f_3d(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const float* rgba, uint32_t w, uint32_t h, uint32_t d,
                              LutTexture3D& out) {
	VkDeviceSize  sz = (VkDeviceSize) w * h * d * 4 * sizeof(float);
	VmaAllocation sa;
	VkBuffer stg = create_and_upload_buffer(alloc, sz, rgba, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);

	VkImageCreateInfo ii{};
	ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ii.imageType     = VK_IMAGE_TYPE_3D;
	ii.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent        = {w, h, d};
	ii.mipLevels     = 1;
	ii.arrayLayers   = 1;
	ii.samples       = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
	ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(alloc, &ii, &ai, &out.image, &out.alloc, nullptr);

	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkBufferImageCopy r{};
	r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	r.imageExtent      = {w, h, d};
	vkCmdCopyBufferToImage(cmd, stg, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	end_one_time_cmd(dev, pool, q, cmd);

	out.view =
	    create_image_view(dev, out.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_VIEW_TYPE_3D);
	out.width  = w;
	out.height = h;
	out.depth  = d;
	vmaDestroyBuffer(alloc, stg, sa);
}

// ============================================================
// Public: OpenPBR LUT upload
// ============================================================
void upload_openpbr_luts(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q) {
	// 3D energy LUTs (32 x 32 x 32)
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize *
		                                    OpenPBR_EnergyTableSize);
		upload_rgba32f_3d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_3d[OpenPBR_LutId_IdealDielectricEnergyComplement]);
	}
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_OpaqueDielectricEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize *
		                                    OpenPBR_EnergyTableSize);
		upload_rgba32f_3d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_3d[OpenPBR_LutId_OpaqueDielectricEnergyComplement]);
	}

	// 2D energy LUTs (32 x 32)
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);
		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx
		        .lut_textures_2d[OpenPBR_LutId_IdealDielectricAverageEnergyComplement]);
	}
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricReflectionRatio_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);
		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealDielectricReflectionRatio]);
	}
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_OpaqueDielectricAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);
		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx
		        .lut_textures_2d[OpenPBR_LutId_OpaqueDielectricAverageEnergyComplement]);
	}
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealMetalEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);
		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealMetalEnergyComplement]);
	}
	// 1D LUT stored as thin 2D (32 x 1)
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealMetalAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize);
		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, 1,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealMetalAverageEnergyComplement]);
	}
	// LTC LUT (32 x 32), float3 → rgba32f
	{
		auto data = float3_lut_to_rgba32f(reinterpret_cast<const float*>(OpenPBR_LTC_Array),
		                                  OpenPBR_LTCTableSize * OpenPBR_LTCTableSize);
		upload_rgba32f_2d(alloc, dev, pool, q, data.data(), OpenPBR_LTCTableSize,
		                  OpenPBR_LTCTableSize,
		                  render_target_ctx.lut_textures_2d[OpenPBR_LutId_LTC]);
	}

	std::cout << "[INFO] Uploaded OpenPBR LUTs\n";
}

// ============================================================
// Public: material texture upload
// ============================================================
void upload_material_textures(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const std::vector<std::shared_ptr<Texture>>& textures) {
	auto& images = render_target_ctx.mat_images;
	auto& allocs = render_target_ctx.mat_allocs;
	auto& views  = render_target_ctx.mat_views;
	images.reserve(textures.size());
	allocs.reserve(textures.size());
	views.reserve(textures.size());

	for (const auto& tex : textures) {
		if (!tex || !tex->valid()) {
			images.push_back(VK_NULL_HANDLE);
			allocs.push_back(VK_NULL_HANDLE);
			views.push_back(VK_NULL_HANDLE);
			continue;
		}
		VkDeviceSize   sz = (VkDeviceSize) tex->width * tex->height * 4;
		VmaAllocation  sa;
		VkBuffer       stg = create_and_upload_buffer(alloc, sz, tex->pixels.data(),
		                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);
		const VkFormat fmt = tex->is_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

		VkImageCreateInfo ii{};
		ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ii.imageType     = VK_IMAGE_TYPE_2D;
		ii.format        = fmt;
		ii.extent        = {(uint32_t) tex->width, (uint32_t) tex->height, 1};
		ii.mipLevels     = 1;
		ii.arrayLayers   = 1;
		ii.samples       = VK_SAMPLE_COUNT_1_BIT;
		ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VmaAllocationCreateInfo ai2{};
		ai2.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage       img;
		VmaAllocation imgalloc;
		if (vmaCreateImage(alloc, &ii, &ai2, &img, &imgalloc, nullptr) != VK_SUCCESS)
			throw std::runtime_error("failed to create material texture");

		VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
		transition_layout(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                  VK_PIPELINE_STAGE_TRANSFER_BIT);
		VkBufferImageCopy r{};
		r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		r.imageExtent      = {(uint32_t) tex->width, (uint32_t) tex->height, 1};
		vkCmdCopyBufferToImage(cmd, stg, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
		transition_layout(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(dev, pool, q, cmd);

		images.push_back(img);
		allocs.push_back(imgalloc);
		views.push_back(create_image_view(dev, img, fmt, VK_IMAGE_VIEW_TYPE_2D));
		vmaDestroyBuffer(alloc, stg, sa);
	}
	std::cout << "[INFO] Uploaded " << textures.size() << " material textures\n";
}
