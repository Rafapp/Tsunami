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

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "slang.h"

namespace {

constexpr float    kPi                  = 3.14159265358979323846f;
constexpr uint64_t kMaxSimulationPixels = 1280ull * 720ull;
struct WaterPushConstantsRaw {
	float    time_seconds                      = 0.0f;
	float    delta_time                        = 1.0f / 60.0f;
	float    propagation                       = 0.18f;
	float    damping                           = 0.028f;
	float    restoring_force                   = 0.08f;
	float    audio_level                       = 0.0f;
	float    height_scale                      = 24.0f;
	float    ripple_radius                     = 0.035f;
	float    impulse_strength                  = 0.0f;
	float    impulse_frequency_hz              = 1.4f;
	float    emitter_u                         = 0.5f;
	float    emitter_v                         = 0.5f;
	uint32_t floating_object_count             = 0;
	uint32_t floating_object_interaction_count = 0;
};

struct FloatingObjectPushConstantsRaw {
	float    time_seconds          = 0.0f;
	float    delta_time            = 1.0f / 60.0f;
	float    height_to_world_scale = 1.0f;
	uint32_t object_count          = 0;
	uint32_t reset_requested       = 0;
};

struct alignas(16) FloatingObjectSettingsGpu {
	glm::vec4 anchor_base_height_yaw{};
	glm::vec4 size_mass{};
	glm::vec4 color_buoyancy{};
	glm::vec4 buoyancy_linear_angular{};
	glm::vec4 motion_planar{};
	glm::vec4 anchor_waterline_yaw{};
};

struct alignas(16) FloatingObjectStateGpu {
	glm::vec4 position_hull{};
	glm::vec4 rotation_pad{};
	glm::vec4 linear_velocity_pad{};
	glm::vec4 angular_velocity_pad{};
};

static_assert(sizeof(WaterPushConstantsRaw) <= 128,
              "Water push constants must fit in Vulkan limits.");
static_assert(sizeof(FloatingObjectPushConstantsRaw) <= 128,
              "Floating object push constants must fit in Vulkan limits.");

const std::array<FloatingObjectSettingsGpu, 3> kDefaultFloatingObjectSettings = {
    FloatingObjectSettingsGpu{
        .anchor_base_height_yaw  = glm::vec4(-0.42f, -0.08f, 0.03f, 12.0f * (kPi / 180.0f)),
        .size_mass               = glm::vec4(0.30f, 0.12f, 0.20f, 1.0f),
        .color_buoyancy          = glm::vec4(0.82f, 0.52f, 0.28f, 34.0f),
        .buoyancy_linear_angular = glm::vec4(7.5f, 1.8f, 13.0f, 5.5f),
        .motion_planar           = glm::vec4(6.5f, 0.35f, 2.5f, 1.4f),
        .anchor_waterline_yaw    = glm::vec4(0.45f, 0.50f, 0.010f, 2.2f),
    },
    FloatingObjectSettingsGpu{
        .anchor_base_height_yaw  = glm::vec4(0.28f, 0.14f, 0.025f, -18.0f * (kPi / 180.0f)),
        .size_mass               = glm::vec4(0.24f, 0.10f, 0.18f, 0.85f),
        .color_buoyancy          = glm::vec4(0.69f, 0.42f, 0.22f, 30.0f),
        .buoyancy_linear_angular = glm::vec4(7.5f, 1.8f, 13.0f, 5.5f),
        .motion_planar           = glm::vec4(6.5f, 0.35f, 2.8f, 1.4f),
        .anchor_waterline_yaw    = glm::vec4(0.45f, 0.44f, 0.010f, 2.2f),
    },
    FloatingObjectSettingsGpu{
        .anchor_base_height_yaw  = glm::vec4(0.06f, 0.38f, 0.04f, 32.0f * (kPi / 180.0f)),
        .size_mass               = glm::vec4(0.36f, 0.14f, 0.24f, 1.25f),
        .color_buoyancy          = glm::vec4(0.91f, 0.66f, 0.35f, 38.0f),
        .buoyancy_linear_angular = glm::vec4(7.5f, 1.8f, 11.0f, 5.5f),
        .motion_planar           = glm::vec4(6.5f, 0.35f, 1.9f, 1.4f),
        .anchor_waterline_yaw    = glm::vec4(0.45f, 0.56f, 0.012f, 2.2f),
    },
};

FloatingObjectStateGpu makeInitialFloatingObjectState(const FloatingObjectSettingsGpu& settings) {
	FloatingObjectStateGpu state{};
	state.position_hull =
	    glm::vec4(settings.anchor_base_height_yaw.x, settings.anchor_base_height_yaw.z,
	              settings.anchor_base_height_yaw.y, 0.0f);
	state.rotation_pad = glm::vec4(0.0f, settings.anchor_base_height_yaw.w, 0.0f, 0.0f);
	return state;
}

float clamp01(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

VkExtent2D computeCappedExtent(VkExtent2D requested_extent, uint64_t max_pixels) {
	if (requested_extent.width == 0 || requested_extent.height == 0) {
		return requested_extent;
	}

	const uint64_t requested_pixels = static_cast<uint64_t>(requested_extent.width) *
	                                  static_cast<uint64_t>(requested_extent.height);
	if (requested_pixels <= max_pixels) {
		return requested_extent;
	}

	const double scale =
	    std::sqrt(static_cast<double>(max_pixels) / static_cast<double>(requested_pixels));
	return VkExtent2D{
	    std::max(1u, static_cast<uint32_t>(
	                     std::lround(static_cast<double>(requested_extent.width) * scale))),
	    std::max(1u, static_cast<uint32_t>(
	                     std::lround(static_cast<double>(requested_extent.height) * scale))),
	};
}

VkExtent2D computeSimulationExtent(VkExtent2D requested_extent) {
	return computeCappedExtent(requested_extent, kMaxSimulationPixels);
}

void createBuffer(VkDevice device, VmaAllocator allocator, VkDeviceSize size,
                  VkBufferUsageFlags usage, VmaMemoryUsage memory_usage, VkBuffer& buffer,
                  VmaAllocation& allocation) {
	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size  = size;
	buffer_info.usage = usage;

	VmaAllocationCreateInfo allocation_info{};
	allocation_info.usage = memory_usage;

	if (vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer, &allocation, nullptr) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to create water simulation buffer");
	}
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

	const SlangResult result      = spCompile(request);
	const char*       diagnostics = spGetDiagnosticOutput(request);
	if (diagnostics != nullptr && diagnostics[0] != '\0') {
		std::cerr << "[SLANG] " << resolved_path << ":\n" << diagnostics << "\n";
	}

	if (result != SLANG_OK) {
		spDestroyCompileRequest(request);
		spDestroySession(session);
		throw std::runtime_error("slang compilation failed: " + resolved_path);
	}

	size_t                spirv_size = 0;
	const void*           spirv_data = spGetEntryPointCode(request, 0, &spirv_size);
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

VkPipeline createComputePipeline(VkDevice device, const std::vector<uint32_t>& spirv,
                                 VkDescriptorSetLayout descriptor_set_layout,
                                 uint32_t push_constant_size, VkPipelineLayout& pipeline_layout) {
	VkShaderModuleCreateInfo module_info{};
	module_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	module_info.codeSize = spirv.size() * sizeof(uint32_t);
	module_info.pCode    = spirv.data();

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(device, &module_info, nullptr, &shader_module) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute shader module");
	}

	VkPushConstantRange push_constant_range{};
	push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	push_constant_range.offset     = 0;
	push_constant_range.size       = push_constant_size;

	VkPipelineLayoutCreateInfo layout_info{};
	layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &descriptor_set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_constant_range;

	if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		vkDestroyShaderModule(device, shader_module, nullptr);
		throw std::runtime_error("failed to create compute pipeline layout");
	}

	VkPipelineShaderStageCreateInfo stage_info{};
	stage_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage_info.module = shader_module;
	stage_info.pName  = "main";

	VkComputePipelineCreateInfo pipeline_info{};
	pipeline_info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeline_info.stage  = stage_info;
	pipeline_info.layout = pipeline_layout;

	VkPipeline pipeline = VK_NULL_HANDLE;
	if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) !=
	    VK_SUCCESS) {
		vkDestroyShaderModule(device, shader_module, nullptr);
		throw std::runtime_error("failed to create compute pipeline");
	}

	vkDestroyShaderModule(device, shader_module, nullptr);
	return pipeline;
}

}        // namespace

namespace simulation {

struct WaterSurfaceSimulation::WaterPushConstants : ::WaterPushConstantsRaw {};
struct WaterSurfaceSimulation::FloatingObjectPushConstants : ::FloatingObjectPushConstantsRaw {};

WaterSurfaceSimulation::WaterSurfaceSimulation(const WaterSurfaceCreateInfo& create_info) :
    m_device(create_info.device),
    m_allocator(create_info.allocator),
    m_output_extent(computeSimulationExtent(create_info.output_extent)),
    m_water_push_constants(new WaterPushConstants()),
    m_object_push_constants(new FloatingObjectPushConstants()) {
	if (m_device == VK_NULL_HANDLE || m_allocator == nullptr || m_output_extent.width == 0 ||
	    m_output_extent.height == 0) {
		throw std::runtime_error(
		    "water surface simulation requires a valid Vulkan device and extent");
	}

	m_floating_object_count = static_cast<uint32_t>(kDefaultFloatingObjectSettings.size());
	m_floating_object_interaction_count = m_floating_object_count;
	createImages();
	createDescriptors();
	createPipeline();
	initializeFloatingObjects();
}

WaterSurfaceSimulation::~WaterSurfaceSimulation() {
	delete m_water_push_constants;
	m_water_push_constants = nullptr;
	delete m_object_push_constants;
	m_object_push_constants = nullptr;

	if (m_device == VK_NULL_HANDLE) {
		return;
	}

	if (m_water_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(m_device, m_water_pipeline, nullptr);
	}
	if (m_object_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(m_device, m_object_pipeline, nullptr);
	}

	if (m_water_pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(m_device, m_water_pipeline_layout, nullptr);
	}
	if (m_object_pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(m_device, m_object_pipeline_layout, nullptr);
	}

	if (m_descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
	}

	if (m_water_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(m_device, m_water_descriptor_set_layout, nullptr);
	}
	if (m_object_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(m_device, m_object_descriptor_set_layout, nullptr);
	}

	if (m_floating_object_settings_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(m_allocator, m_floating_object_settings_buffer,
		                 m_floating_object_settings_allocation);
	}
	if (m_floating_object_states_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(m_allocator, m_floating_object_states_buffer,
		                 m_floating_object_states_allocation);
	}
	if (m_floating_objects_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(m_allocator, m_floating_objects_buffer, m_floating_objects_allocation);
	}
	if (m_floating_object_interactions_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(m_allocator, m_floating_object_interactions_buffer,
		                 m_floating_object_interactions_allocation);
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

const WaterSurfaceDiagnostics&
    WaterSurfaceSimulation::prepareFrame(const WaterSurfaceSettings& settings, float audio_level,
                                         float time_seconds, float delta_time) {
	const float clamped_delta_time = std::clamp(delta_time, 1.0f / 240.0f, 1.0f / 12.0f);
	const float time_scale         = std::clamp(clamped_delta_time * 60.0f, 0.25f, 2.0f);
	const float clamped_audio      = clamp01(audio_level);
	const float gated_audio        = clamp01((clamped_audio - 0.025f) / 0.975f);
	const bool  advance_state =
	    m_last_prepare_time < 0.0f || std::abs(time_seconds - m_last_prepare_time) > 1.0e-5f;

	if (advance_state) {
		const float attack         = std::max(gated_audio - m_previous_audio_level, 0.0f);
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

		const float preview_height_scale =
		    std::clamp(settings.height_scale * (0.84f + m_recent_activity * 0.36f), 0.1f, 28.0f);
		const float max_impulse =
		    std::clamp(0.012f * (16.0f / std::max(preview_height_scale, 16.0f)), 0.0065f, 0.012f);
		m_pending_impulse      = std::min(impulse_strength, max_impulse);
		m_previous_audio_level = gated_audio;
		m_last_prepare_time    = time_seconds;
	}

	const float speed_drive = clamp01(gated_audio * 0.45f + m_recent_activity * 0.85f);
	const float propagation =
	    std::clamp(settings.propagation * (0.70f + speed_drive * 0.30f) * time_scale * time_scale,
	               0.0f, 0.24f);
	const float damping =
	    std::clamp(settings.damping + ((1.0f - speed_drive) * 0.010f) + (speed_drive * 0.004f),
	               0.0060f, 0.090f);
	const float restoring_force = std::clamp(
	    settings.restoring_force * (0.80f + speed_drive * 1.20f) * time_scale * time_scale, 0.0f,
	    0.28f);
	const float orbit_radius = std::clamp(settings.orbit_radius, 0.0f, 0.45f);
	const float orbit_speed  = std::max(settings.orbit_speed, 0.0f);
	const float orbit_angle  = time_seconds * orbit_speed * kPi * 2.0f;
	const float emitter_u    = 0.5f + std::cos(orbit_angle) * orbit_radius;
	const float emitter_v    = 0.5f + std::sin(orbit_angle * 1.618f) * orbit_radius * 0.65f;
	const float ripple_radius =
	    std::clamp(settings.ripple_radius * (0.80f + speed_drive * 0.90f), 0.001f, 0.35f);
	const float height_scale =
	    std::clamp(settings.height_scale * (0.84f + speed_drive * 0.36f), 0.1f, 28.0f);
	const float clamped_emitter_u = std::clamp(emitter_u, 0.05f, 0.95f);
	const float clamped_emitter_v = std::clamp(emitter_v, 0.05f, 0.95f);
	m_height_to_world_scale       = height_scale * 0.045f;

	m_diagnostics.audio_drive_level   = speed_drive;
	m_diagnostics.impulse_strength    = m_pending_impulse;
	m_diagnostics.emitter_u           = clamped_emitter_u;
	m_diagnostics.emitter_v           = clamped_emitter_v;
	m_diagnostics.grid_width          = m_output_extent.width;
	m_diagnostics.grid_height         = m_output_extent.height;
	m_diagnostics.dispatch_groups_x   = (m_output_extent.width + 15) / 16;
	m_diagnostics.dispatch_groups_y   = (m_output_extent.height + 15) / 16;
	m_diagnostics.history_image_count = static_cast<uint32_t>(std::size(m_height_images));
	m_diagnostics.sample_count        = static_cast<uint64_t>(m_output_extent.width) *
	                             static_cast<uint64_t>(m_output_extent.height);
	m_diagnostics.cell_count     = m_output_extent.width > 0 && m_output_extent.height > 0 ?
	                                   static_cast<uint64_t>(m_output_extent.width - 1) *
                                       static_cast<uint64_t>(m_output_extent.height - 1) :
	                                   0ull;
	m_diagnostics.triangle_count = m_diagnostics.cell_count * 2ull;

	m_water_push_constants->time_seconds          = time_seconds;
	m_water_push_constants->delta_time            = clamped_delta_time;
	m_water_push_constants->propagation           = propagation;
	m_water_push_constants->damping               = damping;
	m_water_push_constants->restoring_force       = restoring_force;
	m_water_push_constants->audio_level           = speed_drive;
	m_water_push_constants->height_scale          = height_scale;
	m_water_push_constants->ripple_radius         = ripple_radius;
	m_water_push_constants->impulse_strength      = m_pending_impulse;
	m_water_push_constants->impulse_frequency_hz  = std::max(settings.impulse_frequency_hz, 0.0f);
	m_water_push_constants->emitter_u             = m_diagnostics.emitter_u;
	m_water_push_constants->emitter_v             = m_diagnostics.emitter_v;
	m_water_push_constants->floating_object_count = m_floating_object_count;
	m_water_push_constants->floating_object_interaction_count = m_floating_object_interaction_count;

	m_object_push_constants->time_seconds          = time_seconds;
	m_object_push_constants->delta_time            = clamped_delta_time;
	m_object_push_constants->height_to_world_scale = m_height_to_world_scale;
	m_object_push_constants->object_count          = m_floating_object_count;
	m_object_push_constants->reset_requested       = m_reset_objects_requested ? 1u : 0u;

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

void WaterSurfaceSimulation::requestObjectReset() {
	m_reset_objects_requested = true;
	if (m_object_push_constants != nullptr) {
		m_object_push_constants->reset_requested = 1u;
	}
}

void WaterSurfaceSimulation::initializeFloatingObjects() {
	void* mapped_memory = nullptr;
	if (vmaMapMemory(m_allocator, m_floating_object_settings_allocation, &mapped_memory) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to map floating object settings buffer");
	}
	std::memcpy(mapped_memory, kDefaultFloatingObjectSettings.data(),
	            sizeof(kDefaultFloatingObjectSettings));
	vmaUnmapMemory(m_allocator, m_floating_object_settings_allocation);

	std::array<FloatingObjectStateGpu, kMaxFloatingObjects> initial_states{};
	for (uint32_t index = 0; index < m_floating_object_count; ++index) {
		initial_states[index] =
		    makeInitialFloatingObjectState(kDefaultFloatingObjectSettings[index]);
	}

	if (vmaMapMemory(m_allocator, m_floating_object_states_allocation, &mapped_memory) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to map floating object state buffer");
	}
	std::memcpy(mapped_memory, initial_states.data(), sizeof(initial_states));
	vmaUnmapMemory(m_allocator, m_floating_object_states_allocation);

	std::array<FloatingObjectRenderData, kMaxFloatingObjects> empty_render_data{};
	if (vmaMapMemory(m_allocator, m_floating_objects_allocation, &mapped_memory) != VK_SUCCESS) {
		throw std::runtime_error("failed to map floating object render buffer");
	}
	std::memcpy(mapped_memory, empty_render_data.data(), sizeof(empty_render_data));
	vmaUnmapMemory(m_allocator, m_floating_objects_allocation);

	std::array<FloatingObjectInteractionData, kMaxFloatingObjectInteractions> empty_interactions{};
	if (vmaMapMemory(m_allocator, m_floating_object_interactions_allocation, &mapped_memory) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to map floating object interaction buffer");
	}
	std::memcpy(mapped_memory, empty_interactions.data(), sizeof(empty_interactions));
	vmaUnmapMemory(m_allocator, m_floating_object_interactions_allocation);
}

void WaterSurfaceSimulation::record(VkCommandBuffer command_buffer) {
	VkImageMemoryBarrier output_barrier{};
	output_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	output_barrier.oldLayout =
	    m_output_in_transfer_src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
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
		height_barriers[index].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		height_barriers[index].oldLayout =
		    m_height_layout_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
		height_barriers[index].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
		height_barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		height_barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		height_barriers[index].image               = m_height_images[index];
		height_barriers[index].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		height_barriers[index].srcAccessMask =
		    m_height_layout_initialized ? VK_ACCESS_SHADER_WRITE_BIT : 0;
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

		vkCmdClearColorImage(command_buffer, m_output_image, VK_IMAGE_LAYOUT_GENERAL, &clear_value,
		                     1, &clear_range);
		for (VkImage image : m_height_images) {
			vkCmdClearColorImage(command_buffer, image, VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1,
			                     &clear_range);
		}

		m_clear_history_requested = false;

		VkMemoryBarrier clear_barrier{};
		clear_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		clear_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		clear_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &clear_barrier, 0, nullptr,
		                     0, nullptr);
	}

	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_object_pipeline);
	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
	                        m_object_pipeline_layout, 0, 1,
	                        &m_object_descriptor_sets[m_active_descriptor_set_index], 0, nullptr);
	vkCmdPushConstants(command_buffer, m_object_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                   sizeof(FloatingObjectPushConstants), m_object_push_constants);
	vkCmdDispatch(command_buffer, 1, 1, 1);

	VkMemoryBarrier object_barrier{};
	object_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	object_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	object_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &object_barrier, 0, nullptr, 0,
	                     nullptr);

	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_water_pipeline);
	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_water_pipeline_layout,
	                        0, 1, &m_water_descriptor_sets[m_active_descriptor_set_index], 0,
	                        nullptr);
	m_water_push_constants->floating_object_count             = m_floating_object_count;
	m_water_push_constants->floating_object_interaction_count = m_floating_object_interaction_count;
	vkCmdPushConstants(command_buffer, m_water_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                   sizeof(WaterPushConstants), m_water_push_constants);

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

	m_height_layout_initialized              = true;
	m_output_in_transfer_src                 = true;
	m_active_descriptor_set_index            = 1 - m_active_descriptor_set_index;
	m_reset_objects_requested                = false;
	m_object_push_constants->reset_requested = 0;
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
	createBuffer(m_device, m_allocator,
	             sizeof(FloatingObjectSettingsGpu) * static_cast<VkDeviceSize>(kMaxFloatingObjects),
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
	             m_floating_object_settings_buffer, m_floating_object_settings_allocation);
	createBuffer(m_device, m_allocator,
	             sizeof(FloatingObjectStateGpu) * static_cast<VkDeviceSize>(kMaxFloatingObjects),
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
	             m_floating_object_states_buffer, m_floating_object_states_allocation);
	createBuffer(m_device, m_allocator,
	             sizeof(FloatingObjectRenderData) * static_cast<VkDeviceSize>(kMaxFloatingObjects),
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
	             m_floating_objects_buffer, m_floating_objects_allocation);
	createBuffer(m_device, m_allocator,
	             sizeof(FloatingObjectInteractionData) *
	                 static_cast<VkDeviceSize>(kMaxFloatingObjectInteractions),
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
	             m_floating_object_interactions_buffer, m_floating_object_interactions_allocation);

	VkDescriptorSetLayoutBinding storage_image_binding{};
	storage_image_binding.binding         = 0;
	storage_image_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	storage_image_binding.descriptorCount = 1;
	storage_image_binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding output_binding         = storage_image_binding;
	VkDescriptorSetLayoutBinding current_height_binding = storage_image_binding;
	current_height_binding.binding                      = 1;

	VkDescriptorSetLayoutBinding previous_height_binding = storage_image_binding;
	previous_height_binding.binding                      = 2;

	VkDescriptorSetLayoutBinding storage_buffer_binding{};
	storage_buffer_binding.binding         = 0;
	storage_buffer_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	storage_buffer_binding.descriptorCount = 1;
	storage_buffer_binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding floating_objects_binding = storage_buffer_binding;
	floating_objects_binding.binding                      = 3;

	VkDescriptorSetLayoutBinding floating_object_interactions_binding = floating_objects_binding;
	floating_object_interactions_binding.binding                      = 4;

	const std::array<VkDescriptorSetLayoutBinding, 5> water_bindings = {
	    output_binding,
	    current_height_binding,
	    previous_height_binding,
	    floating_objects_binding,
	    floating_object_interactions_binding,
	};

	VkDescriptorSetLayoutBinding object_height_binding       = storage_image_binding;
	object_height_binding.binding                            = 0;
	VkDescriptorSetLayoutBinding object_settings_binding     = storage_buffer_binding;
	object_settings_binding.binding                          = 1;
	VkDescriptorSetLayoutBinding object_states_binding       = storage_buffer_binding;
	object_states_binding.binding                            = 2;
	VkDescriptorSetLayoutBinding object_render_binding       = storage_buffer_binding;
	object_render_binding.binding                            = 3;
	VkDescriptorSetLayoutBinding object_interactions_binding = storage_buffer_binding;
	object_interactions_binding.binding                      = 4;

	const std::array<VkDescriptorSetLayoutBinding, 5> object_bindings = {
	    object_height_binding, object_settings_binding,     object_states_binding,
	    object_render_binding, object_interactions_binding,
	};

	VkDescriptorSetLayoutCreateInfo layout_info{};
	layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_info.bindingCount = static_cast<uint32_t>(water_bindings.size());
	layout_info.pBindings    = water_bindings.data();

	if (vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr,
	                                &m_water_descriptor_set_layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create water descriptor set layout");
	}

	layout_info.bindingCount = static_cast<uint32_t>(object_bindings.size());
	layout_info.pBindings    = object_bindings.data();
	if (vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr,
	                                &m_object_descriptor_set_layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create floating object descriptor set layout");
	}

	const std::array<VkDescriptorPoolSize, 2> pool_sizes = {
	    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8},
	    VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12},
	};

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets       = 4;
	pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
	pool_info.pPoolSizes    = pool_sizes.data();

	if (vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_descriptor_pool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create water simulation descriptor pool");
	}

	VkDescriptorSetAllocateInfo allocate_info{};
	allocate_info.sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate_info.descriptorPool = m_descriptor_pool;

	const std::array<VkDescriptorSetLayout, 2> water_layouts = {m_water_descriptor_set_layout,
	                                                            m_water_descriptor_set_layout};
	allocate_info.descriptorSetCount = static_cast<uint32_t>(water_layouts.size());
	allocate_info.pSetLayouts        = water_layouts.data();
	if (vkAllocateDescriptorSets(m_device, &allocate_info, m_water_descriptor_sets) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate water descriptor sets");
	}

	const std::array<VkDescriptorSetLayout, 2> object_layouts = {m_object_descriptor_set_layout,
	                                                             m_object_descriptor_set_layout};
	allocate_info.descriptorSetCount = static_cast<uint32_t>(object_layouts.size());
	allocate_info.pSetLayouts        = object_layouts.data();
	if (vkAllocateDescriptorSets(m_device, &allocate_info, m_object_descriptor_sets) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to allocate floating object descriptor sets");
	}

	for (size_t set_index = 0; set_index < std::size(m_water_descriptor_sets); ++set_index) {
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
		const VkDescriptorBufferInfo floating_objects_info = {
		    m_floating_objects_buffer,
		    0,
		    sizeof(FloatingObjectRenderData) * static_cast<VkDeviceSize>(kMaxFloatingObjects),
		};
		const VkDescriptorBufferInfo floating_object_interactions_info = {
		    m_floating_object_interactions_buffer,
		    0,
		    sizeof(FloatingObjectInteractionData) *
		        static_cast<VkDeviceSize>(kMaxFloatingObjectInteractions),
		};

		std::array<VkWriteDescriptorSet, 5> writes{};

		writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet          = m_water_descriptor_sets[set_index];
		writes[0].dstBinding      = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[0].pImageInfo      = &output_info;

		writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet          = m_water_descriptor_sets[set_index];
		writes[1].dstBinding      = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[1].pImageInfo      = &current_height_info;

		writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[2].dstSet          = m_water_descriptor_sets[set_index];
		writes[2].dstBinding      = 2;
		writes[2].descriptorCount = 1;
		writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[2].pImageInfo      = &previous_height_info;

		writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[3].dstSet          = m_water_descriptor_sets[set_index];
		writes[3].dstBinding      = 3;
		writes[3].descriptorCount = 1;
		writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[3].pBufferInfo     = &floating_objects_info;

		writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[4].dstSet          = m_water_descriptor_sets[set_index];
		writes[4].dstBinding      = 4;
		writes[4].descriptorCount = 1;
		writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[4].pBufferInfo     = &floating_object_interactions_info;

		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
		                       nullptr);
	}

	for (size_t set_index = 0; set_index < std::size(m_object_descriptor_sets); ++set_index) {
		const uint32_t current_index = static_cast<uint32_t>(set_index);

		const VkDescriptorImageInfo current_height_info = {
		    VK_NULL_HANDLE,
		    m_height_image_views[current_index],
		    VK_IMAGE_LAYOUT_GENERAL,
		};
		const VkDescriptorBufferInfo settings_info = {
		    m_floating_object_settings_buffer,
		    0,
		    sizeof(FloatingObjectSettingsGpu) * static_cast<VkDeviceSize>(kMaxFloatingObjects),
		};
		const VkDescriptorBufferInfo states_info = {
		    m_floating_object_states_buffer,
		    0,
		    sizeof(FloatingObjectStateGpu) * static_cast<VkDeviceSize>(kMaxFloatingObjects),
		};
		const VkDescriptorBufferInfo render_info = {
		    m_floating_objects_buffer,
		    0,
		    sizeof(FloatingObjectRenderData) * static_cast<VkDeviceSize>(kMaxFloatingObjects),
		};
		const VkDescriptorBufferInfo interaction_info = {
		    m_floating_object_interactions_buffer,
		    0,
		    sizeof(FloatingObjectInteractionData) *
		        static_cast<VkDeviceSize>(kMaxFloatingObjectInteractions),
		};

		std::array<VkWriteDescriptorSet, 5> writes{};
		writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet          = m_object_descriptor_sets[set_index];
		writes[0].dstBinding      = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writes[0].pImageInfo      = &current_height_info;

		writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet          = m_object_descriptor_sets[set_index];
		writes[1].dstBinding      = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[1].pBufferInfo     = &settings_info;

		writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[2].dstSet          = m_object_descriptor_sets[set_index];
		writes[2].dstBinding      = 2;
		writes[2].descriptorCount = 1;
		writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[2].pBufferInfo     = &states_info;

		writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[3].dstSet          = m_object_descriptor_sets[set_index];
		writes[3].dstBinding      = 3;
		writes[3].descriptorCount = 1;
		writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[3].pBufferInfo     = &render_info;

		writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[4].dstSet          = m_object_descriptor_sets[set_index];
		writes[4].dstBinding      = 4;
		writes[4].descriptorCount = 1;
		writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[4].pBufferInfo     = &interaction_info;

		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
		                       nullptr);
	}
}

void WaterSurfaceSimulation::createPipeline() {
	const std::vector<uint32_t> object_spirv =
	    compileSlangShader("shaders/floating_objects.slang", "main");
	m_object_pipeline =
	    createComputePipeline(m_device, object_spirv, m_object_descriptor_set_layout,
	                          sizeof(FloatingObjectPushConstants), m_object_pipeline_layout);

	const std::vector<uint32_t> water_spirv =
	    compileSlangShader("shaders/water_surface.slang", "main");
	m_water_pipeline = createComputePipeline(m_device, water_spirv, m_water_descriptor_set_layout,
	                                         sizeof(WaterPushConstants), m_water_pipeline_layout);
}

}        // namespace simulation
