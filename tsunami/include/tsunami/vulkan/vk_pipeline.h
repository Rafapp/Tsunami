#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "tsunami/ui/selection_panel.h"
#include "tsunami/vulkan/internal/vk_context.h"

// ============================================================
// Push constant structs — one per render mode
//
// All currently share the same 124-byte layout so existing
// Slang shaders compile unchanged When a mode needs new
// fields, replace its alias with a real struct and update
// the corresponding slang file only
// ============================================================

struct PathTracerPushConstants {
	uint32_t  frame                     = 0;
	uint32_t  material_count            = 0;
	int32_t   selected_mesh_index       = -1;
	uint32_t  outline_width             = 1;
	int32_t   debug_view_mode           = 0;
	uint32_t  stage                     = 0;
	uint32_t  spp                       = 1;
	uint32_t  max_bounces               = 8;
	glm::vec4 outline_color             = glm::vec4(1.0f, 0.65f, 0.15f, 1.0f);
	uint32_t  enable_tonemapping        = 1;
	float     exposure_bias             = 2.0f;
	uint32_t  hipr_object_count         = 0;
	uint32_t  hipr_top_k                = HIPR_TOP_K;
	int32_t   hipr_render_rank          = -1;
	uint32_t  hipr_incremental_sort     = 1;
	uint32_t  hipr_clear_order          = 1;
	uint32_t  hipr_vis_enable_tint      = 1;
	uint32_t  hipr_vis_rainbow_tint     = 1;
	uint32_t  hipr_reserved0            = 0;
	uint32_t  hipr_frames_per_object    = 10;
	float     hipr_score_blend          = 0.25f;
	float     hipr_vis_tint_strength    = 0.5f;
	uint32_t  skybox_enabled            = 1;
	uint32_t  directional_light_enabled = 1;
	float     sun_dir_x                 = 0.0f;
	float     sun_dir_y                 = 1.0f;
	float     sun_dir_z                 = 0.0f;
	float     sun_intensity             = 10.0f;
};
static_assert(sizeof(PathTracerPushConstants) == 124);

// Same layout for now
using HiPRPushConstants     = PathTracerPushConstants;
using HiPRVisPushConstants  = PathTracerPushConstants;
using NaivePTPushConstants  = PathTracerPushConstants;
using ObjectIdPushConstants = PathTracerPushConstants;
size_t                                              render_mode_index(ui::RenderDebugViewMode mode);
extern const std::array<ui::RenderDebugViewMode, 4> kRenderModes;
VkDescriptorSetLayoutBinding make_binding(uint32_t binding, VkDescriptorType type,
                                          uint32_t count = 1);
void init_pipeline_modes(VkDevice device, VkDescriptorSetLayout desc_layout);
void destroy_pipeline_modes(VkDevice device);
bool create_pipeline_for_mode(ui::RenderDebugViewMode mode, VkPipeline& out_pipeline,
                              size_t* out_shader_size_bytes = nullptr);
bool ensure_pipeline_for_mode(ui::RenderDebugViewMode mode);
bool build_all_mode_pipelines();
bool rebuild_pipeline(ui::RenderDebugViewMode mode);
