#include "tsunami/simulation/water_surface_simulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "slang.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct PushConstants {
	float time_seconds         = 0.0f;
	float delta_time           = 1.0f / 60.0f;
	float propagation          = 0.18f;
	float damping              = 0.028f;
	float restoring_force      = 0.08f;
	float audio_level          = 0.0f;
	float height_scale         = 24.0f;
	float ripple_radius        = 0.035f;
	float impulse_strength     = 0.0f;
	float impulse_frequency_hz = 1.4f;
	float emitter_u            = 0.5f;
	float emitter_v            = 0.5f;
};

static_assert(sizeof(PushConstants) <= 128, "Water push constants must fit in Vulkan limits.");

float clamp01(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

std::string resolveShaderPath(const std::string& relative_path) {
	namespace fs = std::filesystem;

	const std::array<fs::path, 5> candidates = {
	    fs::path(relative_path),
	    fs::path("tsunami") / relative_path,
	    fs::path("bin") / relative_path,
	    fs::path("build/bin") / relative_path,
	    fs::path("build-debug/bin") / relative_path,
	};

	for (const fs::path& candidate : candidates) {
		if (fs::exists(candidate)) {
			return candidate.string();
		}
	}

	throw std::runtime_error("could not find shader source: " + relative_path);
}

std::vector<uint32_t> compileSlangShader(const std::string& path, const std::string& entry_point) {
	const std::string resolved_path = resolveShaderPath(path);

	SlangSession*        session = spCreateSession(nullptr);
	SlangCompileRequest* request = spCreateCompileRequest(session);

	const int target_index = spAddCodeGenTarget(request, SLANG_SPIRV);
	spSetTargetProfile(request, target_index, spFindProfile(session, "spirv_1_3"));

	const int unit_index = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
	spAddTranslationUnitSourceFile(request, unit_index, resolved_path.c_str());
	spAddEntryPoint(request, unit_index, entry_point.c_str(), SLANG_STAGE_COMPUTE);

	const SlangResult result = spCompile(request);
	const char*       diagnostics = spGetDiagnosticOutput(request);
	if (diagnostics != nullptr && diagnostics[0] != '\0') {
		std::cerr << "[SLANG] " << resolved_path << ":\n" << diagnostics << "\n";
	}

	if (result != SLANG_OK) {
		spDestroyCompileRequest(request);
		spDestroySession(session);
		throw std::runtime_error("slang compilation failed: " + resolved_path);
	}

	size_t      spirv_size = 0;
	const void* spirv_data = spGetEntryPointCode(request, 0, &spirv_size);
	std::vector<uint32_t> spirv(spirv_size / sizeof(uint32_t));
	std::memcpy(spirv.data(), spirv_data, spirv_size);

	spDestroyCompileRequest(request);
	spDestroySession(session);

	return spirv;
}

void createImage(VkDevice device, VmaAllocator allocator, VkExtent2D extent, VkFormat format,
                 VkImageUsageFlags usage, VkImage& image, VmaAllocation& allocation) {
	VkImageCreateInfo image_info{};
	image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType     = VK_IMAGE_TYPE_2D;
	image_info.format        = format;
	image_info.extent        = {extent.width, extent.height, 1};
	image_info.mipLevels     = 1;
	image_info.arrayLayers   = 1;
	image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
	image_info.usage         = usage;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocation_info{};
	allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	if (vmaCreateImage(allocator, &image_info, &allocation_info, &image, &allocation, nullptr) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to create water simulation image");
	}
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format) {
	VkImageViewCreateInfo view_info{};
	view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image                           = image;
	view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format                          = format;
	view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	view_info.subresourceRange.baseMipLevel   = 0;
	view_info.subresourceRange.levelCount     = 1;
	view_info.subresourceRange.baseArrayLayer = 0;
	view_info.subresourceRange.layerCount     = 1;

	VkImageView image_view = VK_NULL_HANDLE;
	if (vkCreateImageView(device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
		throw std::runtime_error("failed to create water simulation image view");
	}

	return image_view;
}

}        // namespace

namespace simulation {

struct WaterSurfaceSimulation::PushConstants : ::PushConstants {};

WaterSurfaceSimulation::WaterSurfaceSimulation(const WaterSurfaceCreateInfo& create_info)
    : m_device(create_info.device), m_allocator(create_info.allocator),
      m_output_extent(create_info.output_extent), m_push_constants(new PushConstants()) {
	if (m_device == VK_NULL_HANDLE || m_allocator == nullptr || m_output_extent.width == 0 ||
	    m_output_extent.height == 0) {
		throw std::runtime_error("water surface simulation requires a valid Vulkan device and extent");
	}

	createImages();
	createDescriptors();
	createPipeline();
}

WaterSurfaceSimulation::~WaterSurfaceSimulation() {
	delete m_push_constants;
	m_push_constants = nullptr;

	if (m_device == VK_NULL_HANDLE) {
		return;
	}

	if (m_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(m_device, m_pipeline, nullptr);
	}

	if (m_pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
	}

	if (m_descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
	}

	if (m_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
	}

	for (VkImageView image_view : m_height_image_views) {
		if (image_view != VK_NULL_HANDLE) {
			vkDestroyImageView(m_device, image_view, nullptr);
		}
	}

	if (m_output_image_view != VK_NULL_HANDLE) {
		vkDestroyImageView(m_device, m_output_image_view, nullptr);
	}

	for (size_t index = 0; index < std::size(m_height_images); ++index) {
		if (m_height_images[index] != VK_NULL_HANDLE) {
			vmaDestroyImage(m_allocator, m_height_images[index], m_height_allocations[index]);
		}
	}

	if (m_output_image != VK_NULL_HANDLE) {
		vmaDestroyImage(m_allocator, m_output_image, m_output_allocation);
	}
}

const WaterSurfaceDiagnostics& WaterSurfaceSimulation::prepareFrame(
    const WaterSurfaceSettings& settings, float audio_level, float time_seconds, float delta_time) {
	const float clamped_delta_time = std::clamp(delta_time, 1.0f / 240.0f, 1.0f / 12.0f);
	const float time_scale         = std::clamp(clamped_delta_time * 60.0f, 0.25f, 2.0f);
	const float clamped_audio      = clamp01(audio_level);
	const float gated_audio        = clamp01((clamped_audio - 0.025f) / 0.975f);
	const bool  advance_state      = m_last_prepare_time < 0.0f ||
	                            std::abs(time_seconds - m_last_prepare_time) > 1.0e-5f;

	if (advance_state) {
		const float attack = std::max(gated_audio - m_previous_audio_level, 0.0f);
		const float activity_decay = std::exp(-clamped_delta_time * 4.0f);
		m_recent_activity          = std::max(gated_audio, m_recent_activity * activity_decay);

		float impulse_strength = 0.0f;
		if (gated_audio > 0.0f) {
			const float speed_drive = clamp01(gated_audio * 0.55f + m_recent_activity * 0.75f);
			const float emission_rate =
			    std::max(settings.impulse_frequency_hz, 0.0f) * (0.18f + speed_drive * 1.55f);
			m_emission_accumulator += clamped_delta_time * emission_rate;

			const bool cadence_pulse = m_emission_accumulator >= 1.0f;
			const bool attack_pulse  = attack > 0.040f;
			if (cadence_pulse) {
				m_emission_accumulator -= std::floor(m_emission_accumulator);
			}

			if (cadence_pulse || attack_pulse) {
				const float energy = clamp01(speed_drive * 0.80f + attack * 2.00f);
				impulse_strength   = std::max(settings.base_impulse, 0.0f) +
				                   energy * std::max(settings.audio_impulse_scale, 0.0f) *
				                       (0.70f + speed_drive * 0.90f);
			}
		} else {
			m_emission_accumulator = 0.0f;
		}

		m_pending_impulse     = impulse_strength;
		m_previous_audio_level = gated_audio;
		m_last_prepare_time    = time_seconds;
	}

	const float speed_drive = clamp01(gated_audio * 0.45f + m_recent_activity * 0.85f);
	const float propagation =
	    std::clamp(settings.propagation * (0.55f + speed_drive * 0.85f) * time_scale * time_scale,
	               0.0f, 0.47f);
	const float damping =
	    std::clamp(settings.damping + ((1.0f - speed_drive) * 0.0075f) -
	                   (speed_drive * 0.0035f),
	               0.0010f, 0.080f);
	const float restoring_force =
	    std::clamp(settings.restoring_force * (0.80f + speed_drive * 1.20f) *
	                   time_scale * time_scale,
	               0.0f, 0.35f);
	const float orbit_radius     = std::clamp(settings.orbit_radius, 0.0f, 0.45f);
	const float orbit_speed      = std::max(settings.orbit_speed, 0.0f);
	const float orbit_angle      = time_seconds * orbit_speed * kPi * 2.0f;
	const float emitter_u        = 0.5f + std::cos(orbit_angle) * orbit_radius;
	const float emitter_v        = 0.5f + std::sin(orbit_angle * 1.618f) * orbit_radius * 0.65f;
	const float ripple_radius =
	    std::clamp(settings.ripple_radius * (0.80f + speed_drive * 0.90f), 0.001f, 0.35f);
	const float height_scale =
	    std::max(settings.height_scale * (0.82f + speed_drive * 0.78f), 0.1f);

	m_diagnostics.audio_drive_level = speed_drive;
	m_diagnostics.impulse_strength  = m_pending_impulse;
	m_diagnostics.emitter_u         = std::clamp(emitter_u, 0.05f, 0.95f);
	m_diagnostics.emitter_v         = std::clamp(emitter_v, 0.05f, 0.95f);

	m_push_constants->time_seconds         = time_seconds;
	m_push_constants->delta_time           = clamped_delta_time;
	m_push_constants->propagation          = propagation;
	m_push_constants->damping              = damping;
	m_push_constants->restoring_force      = restoring_force;
	m_push_constants->audio_level          = speed_drive;
	m_push_constants->height_scale         = height_scale;
	m_push_constants->ripple_radius        = ripple_radius;
	m_push_constants->impulse_strength     = m_pending_impulse;
	m_push_constants->impulse_frequency_hz = std::max(settings.impulse_frequency_hz, 0.0f);
	m_push_constants->emitter_u            = m_diagnostics.emitter_u;
	m_push_constants->emitter_v            = m_diagnostics.emitter_v;

	return m_diagnostics;
}

void WaterSurfaceSimulation::requestReset() {
	m_clear_history_requested = true;
	m_previous_audio_level    = 0.0f;
	m_recent_activity         = 0.0f;
	m_emission_accumulator    = 0.0f;
	m_pending_impulse         = 0.0f;
	m_last_prepare_time       = -1.0f;
}

void WaterSurfaceSimulation::record(VkCommandBuffer command_buffer) {
	VkImageMemoryBarrier output_barrier{};
	output_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	output_barrier.oldLayout           = m_output_in_transfer_src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
	                                                              : VK_IMAGE_LAYOUT_UNDEFINED;
	output_barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	output_barrier.image               = m_output_image;
	output_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	output_barrier.srcAccessMask       = m_output_in_transfer_src ? VK_ACCESS_TRANSFER_READ_BIT : 0;
	output_barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

	vkCmdPipelineBarrier(command_buffer,
	                     m_output_in_transfer_src ? VK_PIPELINE_STAGE_TRANSFER_BIT :
	                                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
	                     &output_barrier);

	std::array<VkImageMemoryBarrier, 2> height_barriers{};
	for (size_t index = 0; index < height_barriers.size(); ++index) {
		height_barriers[index].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		height_barriers[index].oldLayout           = m_height_layout_initialized ?
		                                                 VK_IMAGE_LAYOUT_GENERAL :
		                                                 VK_IMAGE_LAYOUT_UNDEFINED;
		height_barriers[index].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
		height_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		height_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		height_barriers[index].image               = m_height_images[index];
		height_barriers[index].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		height_barriers[index].srcAccessMask = m_height_layout_initialized ? VK_ACCESS_SHADER_WRITE_BIT : 0;
		height_barriers[index].dstAccessMask =
		    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	}

	vkCmdPipelineBarrier(command_buffer,
	                     m_height_layout_initialized ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
	                                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
	                     static_cast<uint32_t>(height_barriers.size()), height_barriers.data());

	if (m_clear_history_requested) {
		const VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		const VkClearColorValue       clear_value = {};

		vkCmdClearColorImage(command_buffer, m_output_image, VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1,
		                     &clear_range);
		for (VkImage image : m_height_images) {
			vkCmdClearColorImage(command_buffer, image, VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1,
			                     &clear_range);
		}

		m_clear_history_requested = false;
	}

	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 0, 1,
	                        &m_descriptor_sets[m_active_descriptor_set_index], 0, nullptr);
	vkCmdPushConstants(command_buffer, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                   sizeof(PushConstants), m_push_constants);

	const uint32_t group_x = (m_output_extent.width + 15) / 16;
	const uint32_t group_y = (m_output_extent.height + 15) / 16;
	vkCmdDispatch(command_buffer, group_x, group_y, 1);

	VkImageMemoryBarrier transfer_barrier{};
	transfer_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	transfer_barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
	transfer_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	transfer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transfer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	transfer_barrier.image               = m_output_image;
	transfer_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	transfer_barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
	transfer_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
	                     &transfer_barrier);

	m_height_layout_initialized   = true;
	m_output_in_transfer_src      = true;
	m_active_descriptor_set_index = 1 - m_active_descriptor_set_index;
}

void WaterSurfaceSimulation::createImages() {
	createImage(m_device, m_allocator, m_output_extent, VK_FORMAT_R8G8B8A8_UNORM,
	            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	                VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	            m_output_image, m_output_allocation);
	m_output_image_view = createImageView(m_device, m_output_image, VK_FORMAT_R8G8B8A8_UNORM);

	for (size_t index = 0; index < std::size(m_height_images); ++index) {
		createImage(m_device, m_allocator, m_output_extent, VK_FORMAT_R32_SFLOAT,
		            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		            m_height_images[index], m_height_allocations[index]);
		m_height_image_views[index] =
		    createImageView(m_device, m_height_images[index], VK_FORMAT_R32_SFLOAT);
	}
}

void WaterSurfaceSimulation::createDescriptors() {
	VkDescriptorSetLayoutBinding output_binding{};
	output_binding.binding         = 0;
	output_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	output_binding.descriptorCount = 1;
	output_binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding current_height_binding = output_binding;
	current_height_binding.binding                      = 1;

	VkDescriptorSetLayoutBinding previous_height_binding = output_binding;
	previous_height_binding.binding                      = 2;

	const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
	    output_binding,
	    current_height_binding,
	    previous_height_binding,
	};

	VkDescriptorSetLayoutCreateInfo layout_info{};
	layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();

	if (vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr, &m_descriptor_set_layout) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to create water simulation descriptor set layout");
	}

	VkDescriptorPoolSize pool_size{};
	pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	pool_size.descriptorCount = 6;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets       = 2;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;

	if (vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_descriptor_pool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create water simulation descriptor pool");
	}

	const std::array<VkDescriptorSetLayout, 2> layouts = {m_descriptor_set_layout,
	                                                      m_descriptor_set_layout};

	VkDescriptorSetAllocateInfo allocate_info{};
	allocate_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate_info.descriptorPool     = m_descriptor_pool;
	allocate_info.descriptorSetCount = static_cast<uint32_t>(layouts.size());
	allocate_info.pSetLayouts        = layouts.data();

	if (vkAllocateDescriptorSets(m_device, &allocate_info, m_descriptor_sets) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate water simulation descriptor sets");
	}

	for (size_t set_index = 0; set_index < std::size(m_descriptor_sets); ++set_index) {
		const uint32_t current_index  = static_cast<uint32_t>(set_index);
		const uint32_t previous_index = 1u - current_index;

		const VkDescriptorImageInfo output_info = {
		    VK_NULL_HANDLE,
		    m_output_image_view,
		    VK_IMAGE_LAYOUT_GENERAL,
		};
		const VkDescriptorImageInfo current_height_info = {
		    VK_NULL_HANDLE,
		    m_height_image_views[current_index],
		    VK_IMAGE_LAYOUT_GENERAL,
		};
		const VkDescriptorImageInfo previous_height_info = {
		    VK_NULL_HANDLE,
		    m_height_image_views[previous_index],
		    VK_IMAGE_LAYOUT_GENERAL,
		};

		std::array<VkWriteDescriptorSet, 3> writes{};

		writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet          = m_descriptor_sets[set_index];
		writes[0].dstBinding      = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[0].pImageInfo      = &output_info;

		writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet          = m_descriptor_sets[set_index];
		writes[1].dstBinding      = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[1].pImageInfo      = &current_height_info;

		writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[2].dstSet          = m_descriptor_sets[set_index];
		writes[2].dstBinding      = 2;
		writes[2].descriptorCount = 1;
		writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[2].pImageInfo      = &previous_height_info;

		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
		                       nullptr);
	}
}

void WaterSurfaceSimulation::createPipeline() {
	const std::vector<uint32_t> spirv = compileSlangShader("shaders/water_surface.slang", "main");

	VkShaderModuleCreateInfo module_info{};
	module_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	module_info.codeSize = spirv.size() * sizeof(uint32_t);
	module_info.pCode    = spirv.data();

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(m_device, &module_info, nullptr, &shader_module) != VK_SUCCESS) {
		throw std::runtime_error("failed to create water simulation shader module");
	}

	VkPushConstantRange push_constant_range{};
	push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	push_constant_range.offset     = 0;
	push_constant_range.size       = sizeof(PushConstants);

	VkPipelineLayoutCreateInfo layout_info{};
	layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &m_descriptor_set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_constant_range;

	if (vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipeline_layout) !=
	    VK_SUCCESS) {
		vkDestroyShaderModule(m_device, shader_module, nullptr);
		throw std::runtime_error("failed to create water simulation pipeline layout");
	}

	VkPipelineShaderStageCreateInfo stage_info{};
	stage_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage_info.module = shader_module;
	stage_info.pName  = "main";

	VkComputePipelineCreateInfo pipeline_info{};
	pipeline_info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeline_info.stage  = stage_info;
	pipeline_info.layout = m_pipeline_layout;

	if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
	                             &m_pipeline) != VK_SUCCESS) {
		vkDestroyShaderModule(m_device, shader_module, nullptr);
		throw std::runtime_error("failed to create water simulation pipeline");
	}

	vkDestroyShaderModule(m_device, shader_module, nullptr);
}

}        // namespace simulation
