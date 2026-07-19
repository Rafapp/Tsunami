#include "tsunami/ui/selection_panel.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>
#include <string_view>
#include <utility>

namespace ui {

SelectionContext::SelectionContext() {
	editor_material = Material{}.pack();
}

SelectionContext selection_ctx{};

constexpr float kEditorTranslationMin = -2.0f;
constexpr float kEditorTranslationMax = 2.0f;
constexpr float kEditorRotationMinDeg = -180.0f;
constexpr float kEditorRotationMaxDeg = 180.0f;

struct ChameleonMaterialTransition {
	std::array<uint8_t, 3> start_color_rgb255;
	std::array<uint8_t, 3> end_color_rgb255;
	float                  start_metalness;
	float                  end_metalness;
	float                  start_roughness;
	float                  end_roughness;
	float                  start_transmission;
	float                  end_transmission;
};

// Colors are kept in the same 0-255 representation shown by the GUI and normalized only
// when they are written to the renderer's material structure.
const std::array<ChameleonMaterialTransition, 3> kChameleonMaterialTransitions = {{
    {{{255, 255, 255}}, {{71, 77, 50}}, 0.0f, 0.0f, 0.5f, 0.2f, 0.0f, 0.0f},
    {{{255, 255, 255}}, {{77, 32, 19}}, 0.0f, 0.0f, 0.482f, 0.5f, 0.0f, 0.0f},
    {{{255, 255, 255}}, {{85, 85, 85}}, 0.0f, 1.0f, 0.414f, 0.347f, 0.0f, 0.0f},
}};

static glm::vec3 normalizeRgb255(const std::array<uint8_t, 3>& color) {
	return glm::vec3(static_cast<float>(color[0]), static_cast<float>(color[1]),
	                 static_cast<float>(color[2])) /
	       255.0f;
}

static glm::vec3 interpolateChameleonColor(const ChameleonMaterialTransition& transition,
                                           float                              level) {
	const glm::vec3 start_color = normalizeRgb255(transition.start_color_rgb255);
	const glm::vec3 end_color   = normalizeRgb255(transition.end_color_rgb255);
	return start_color + level * (end_color - start_color);
}

static bool containsCaseInsensitive(std::string_view value, std::string_view needle) {
	if (needle.empty() || value.size() < needle.size()) {
		return false;
	}

	const auto match = std::search(value.begin(), value.end(), needle.begin(), needle.end(),
	                               [](unsigned char lhs, unsigned char rhs) {
		                               return std::tolower(lhs) == std::tolower(rhs);
	                               });
	return match != value.end();
}

static int chameleonPresetIndexFromName(std::string_view name) {
	if (containsCaseInsensitive(name, "chameleon.002")) {
		return 2;
	}
	if (containsCaseInsensitive(name, "chameleon.001")) {
		return 1;
	}
	if (containsCaseInsensitive(name, "chameleon")) {
		return 0;
	}
	return -1;
}

static int selectedChameleonPresetIndex(const Scene* scene) {
	if (scene == nullptr || selection_ctx.selected_mesh_index < 0 ||
	    selection_ctx.selected_mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		return -1;
	}

	const int name_preset =
	    chameleonPresetIndexFromName(meshDisplayName(scene, selection_ctx.selected_mesh_index));
	if (name_preset >= 0) {
		return name_preset;
	}

	const Mesh* selected_mesh = scene->m_meshes[selection_ctx.selected_mesh_index].get();
	return selected_mesh != nullptr ? chameleonPresetIndexFromName(selected_mesh->m_material_name) :
	                                  -1;
}

std::string meshDisplayName(const Scene* scene, int mesh_index) {
	if (scene == nullptr || mesh_index < 0 ||
	    mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		return "None";
	}

	const Mesh* mesh = scene->m_meshes[mesh_index].get();
	if (mesh == nullptr || mesh->m_name.empty()) {
		return "Mesh " + std::to_string(mesh_index);
	}

	return mesh->m_name;
}

void rebuildObjectIdMap(const Scene* scene) {
	selection_ctx.object_id_map.clear();
	if (scene == nullptr) {
		return;
	}

	selection_ctx.object_id_map.reserve(scene->m_meshes.size());
	for (int mesh_index = 0; mesh_index < static_cast<int>(scene->m_meshes.size()); ++mesh_index) {
		const auto&   mesh = scene->m_meshes[mesh_index];
		ObjectIdEntry entry{};
		entry.object_id      = mesh_index;
		entry.mesh_index     = mesh_index;
		entry.material_index = mesh_index;
		entry.display_name   = meshDisplayName(scene, mesh_index);
		if (mesh == nullptr || mesh->m_material == nullptr) {
			entry.material_index = -1;
		}
		selection_ctx.object_id_map.push_back(std::move(entry));
	}
}

static void refreshSelectedMaterialEditor(const Scene* scene) {
	if (scene == nullptr || selection_ctx.selected_mesh_index < 0 ||
	    selection_ctx.selected_mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		selection_ctx.editor_material = Material{}.pack();
		return;
	}

	const auto& mesh              = scene->m_meshes[selection_ctx.selected_mesh_index];
	selection_ctx.editor_material = (mesh != nullptr && mesh->m_material != nullptr) ?
	                                    mesh->m_material->pack() :
	                                    Material{}.pack();
}

static uint32_t sanitizeTopKValue(uint64_t value) {
	if (value < 1ull) {
		return 1u;
	}
	if (value > 16ull) {
		return 16u;
	}
	return static_cast<uint32_t>(value);
}

static uint32_t sanitizeTenStepFrameValue(uint64_t value) {
	// Allowed set: 1, 10, 20, ... 100
	if (value <= 1ull) {
		return 1u;
	}
	if (value >= 100ull) {
		return 100u;
	}

	uint32_t clamped = static_cast<uint32_t>(value);
	uint32_t snapped = ((clamped + 5u) / 10u) * 10u;        // nearest 10
	snapped          = std::max(snapped, 10u);
	snapped          = std::min(snapped, 100u);
	return snapped;
}

static glm::vec3 rainbowColorFromLoudness(float loudness) {
	const float level = std::clamp(loudness, 0.0f, 1.0f);
	float       r     = 0.0f;
	float       g     = 0.0f;
	float       b     = 0.0f;

	ImGui::ColorConvertHSVtoRGB((1.0f - level) * 0.75f, 0.95f, 1.0f, r, g, b);
	return glm::vec3(r, g, b);
}

static float voiceDrivenScalarValue(VoiceDrivenParameter parameter, float loudness) {
	const float level = std::clamp(loudness, 0.0f, 1.0f);
	switch (parameter) {
		case VoiceDrivenParameter::Metalness:
			return level;
		case VoiceDrivenParameter::Roughness:
			return 0.02f + level * (1.0f - 0.02f);
		case VoiceDrivenParameter::Transmission:
			return level;
		case VoiceDrivenParameter::Ior:
			return 1.0f + level * (2.5f - 1.0f);
		case VoiceDrivenParameter::EmissionIntensity:
			return level * 20.0f;
		case VoiceDrivenParameter::ObjectScale:
			return 0.25f + level * (2.0f - 0.25f);
		case VoiceDrivenParameter::ObjectTranslateX:
		case VoiceDrivenParameter::ObjectTranslateY:
		case VoiceDrivenParameter::ObjectTranslateZ:
			return kEditorTranslationMin + level * (kEditorTranslationMax - kEditorTranslationMin);
		case VoiceDrivenParameter::ObjectRotateX:
		case VoiceDrivenParameter::ObjectRotateY:
		case VoiceDrivenParameter::ObjectRotateZ:
			return kEditorRotationMinDeg + level * (kEditorRotationMaxDeg - kEditorRotationMinDeg);
		case VoiceDrivenParameter::ChameleonMaterial:
		case VoiceDrivenParameter::BaseTint:
		case VoiceDrivenParameter::EmissionColor:
			return level;
	}

	return level;
}

static const char* voiceDrivenParameterLabel(VoiceDrivenParameter parameter) {
	switch (parameter) {
		case VoiceDrivenParameter::BaseTint:
			return "Base tint";
		case VoiceDrivenParameter::EmissionColor:
			return "Emission color";
		case VoiceDrivenParameter::Metalness:
			return "Metalness";
		case VoiceDrivenParameter::Roughness:
			return "Roughness";
		case VoiceDrivenParameter::Transmission:
			return "Transmission";
		case VoiceDrivenParameter::Ior:
			return "IOR";
		case VoiceDrivenParameter::EmissionIntensity:
			return "Emission intensity";
		case VoiceDrivenParameter::ObjectScale:
			return "Object scale";
		case VoiceDrivenParameter::ObjectTranslateX:
			return "Object translation X (world)";
		case VoiceDrivenParameter::ObjectTranslateY:
			return "Object translation Y (world)";
		case VoiceDrivenParameter::ObjectTranslateZ:
			return "Object translation Z (world)";
		case VoiceDrivenParameter::ObjectRotateX:
			return "Object rotation X";
		case VoiceDrivenParameter::ObjectRotateY:
			return "Object rotation Y";
		case VoiceDrivenParameter::ObjectRotateZ:
			return "Object rotation Z";
		case VoiceDrivenParameter::ChameleonMaterial:
			return "Chameleon material (all fields)";
	}

	return "Unknown";
}

static bool voiceParameterUsesRainbowColor(VoiceDrivenParameter parameter) {
	return parameter == VoiceDrivenParameter::BaseTint ||
	       parameter == VoiceDrivenParameter::EmissionColor;
}

struct VoiceDrivenSyncResult {
	bool material_changed  = false;
	bool transform_changed = false;
};

static VoiceDrivenSyncResult syncVoiceDrivenSelectionParameter(float loudness) {
	VoiceDrivenSyncResult result{};
	if (selection_ctx.material_edit_mode != MaterialEditMode::Voice ||
	    selection_ctx.selected_mesh_index < 0) {
		return result;
	}

	switch (selection_ctx.voice_parameter) {
		case VoiceDrivenParameter::BaseTint: {
			const glm::vec3 target_color = rainbowColorFromLoudness(loudness);
			const glm::vec3 current      = glm::vec3(selection_ctx.editor_material.base_color);
			if (glm::length(current - target_color) <= 1.0e-4f) {
				return result;
			}
			selection_ctx.editor_material.base_color =
			    glm::vec4(target_color, selection_ctx.editor_material.base_color.a);
			result.material_changed = true;
			return result;
		}
		case VoiceDrivenParameter::EmissionColor: {
			const glm::vec3 target_color = rainbowColorFromLoudness(loudness);
			const glm::vec3 current      = glm::vec3(selection_ctx.editor_material.emission_color);
			if (glm::length(current - target_color) <= 1.0e-4f) {
				return result;
			}
			selection_ctx.editor_material.emission_color =
			    glm::vec4(target_color, selection_ctx.editor_material.emission_color.a);
			result.material_changed = true;
			return result;
		}
		case VoiceDrivenParameter::Metalness: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			if (std::abs(selection_ctx.editor_material.base_metalness - target_value) <= 1.0e-4f) {
				return result;
			}
			selection_ctx.editor_material.base_metalness = target_value;
			result.material_changed                      = true;
			return result;
		}
		case VoiceDrivenParameter::Roughness: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			if (std::abs(selection_ctx.editor_material.specular_roughness - target_value) <=
			    1.0e-4f) {
				return result;
			}
			selection_ctx.editor_material.specular_roughness = target_value;
			result.material_changed                          = true;
			return result;
		}
		case VoiceDrivenParameter::Transmission: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			if (std::abs(selection_ctx.editor_material.transmission_weight - target_value) <=
			    1.0e-4f) {
				return result;
			}
			selection_ctx.editor_material.transmission_weight = target_value;
			result.material_changed                           = true;
			return result;
		}
		case VoiceDrivenParameter::Ior: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			if (std::abs(selection_ctx.editor_material.specular_ior - target_value) <= 1.0e-4f) {
				return result;
			}
			selection_ctx.editor_material.specular_ior = target_value;
			result.material_changed                    = true;
			return result;
		}
		case VoiceDrivenParameter::EmissionIntensity: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			if (std::abs(selection_ctx.editor_material.emission_luminance - target_value) <=
			    1.0e-4f) {
				return result;
			}
			selection_ctx.editor_material.emission_luminance = target_value;
			result.material_changed                          = true;
			return result;
		}
		case VoiceDrivenParameter::ObjectScale: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			if (std::abs(selection_ctx.editor_scale - target_value) <= 1.0e-4f) {
				return result;
			}
			selection_ctx.editor_scale = target_value;
			result.transform_changed   = true;
			return result;
		}
		case VoiceDrivenParameter::ObjectTranslateX:
		case VoiceDrivenParameter::ObjectTranslateY:
		case VoiceDrivenParameter::ObjectTranslateZ: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			const int axis = static_cast<int>(selection_ctx.voice_parameter) -
			                 static_cast<int>(VoiceDrivenParameter::ObjectTranslateX);
			if (axis < 0 || axis > 2 ||
			    std::abs(selection_ctx.editor_translation[axis] - target_value) <= 1.0e-4f) {
				return result;
			}
			selection_ctx.editor_translation[axis] = target_value;
			result.transform_changed               = true;
			return result;
		}
		case VoiceDrivenParameter::ObjectRotateX:
		case VoiceDrivenParameter::ObjectRotateY:
		case VoiceDrivenParameter::ObjectRotateZ: {
			const float target_value =
			    voiceDrivenScalarValue(selection_ctx.voice_parameter, loudness);
			const int axis = static_cast<int>(selection_ctx.voice_parameter) -
			                 static_cast<int>(VoiceDrivenParameter::ObjectRotateX);
			if (axis < 0 || axis > 2 ||
			    std::abs(selection_ctx.editor_rotation_deg[axis] - target_value) <= 1.0e-4f) {
				return result;
			}
			selection_ctx.editor_rotation_deg[axis] = target_value;
			result.transform_changed                = true;
			return result;
		}
		case VoiceDrivenParameter::ChameleonMaterial:
			// The selected-chameleon composite path owns this target and updates all
			// requested material fields together.
			return result;
	}

	return result;
}

bool selectMesh(const Scene* scene, int mesh_index) {
	const int max_mesh_index =
	    (scene != nullptr) ? static_cast<int>(scene->m_meshes.size()) - 1 : -1;
	const int clamped_index = (mesh_index >= 0 && mesh_index <= max_mesh_index) ? mesh_index : -1;
	if (selection_ctx.selected_mesh_index == clamped_index) {
		return false;
	}

	selection_ctx.selected_mesh_index = clamped_index;
	refreshSelectedMaterialEditor(scene);
	if (selectedChameleonPresetIndex(scene) >= 0) {
		selection_ctx.voice_parameter = VoiceDrivenParameter::ChameleonMaterial;
	} else if (selection_ctx.voice_parameter == VoiceDrivenParameter::ChameleonMaterial) {
		selection_ctx.voice_parameter = VoiceDrivenParameter::BaseTint;
	}
	if (clamped_index < 0) {
		selection_ctx.editor_scale        = 1.0f;
		selection_ctx.editor_translation  = glm::vec3(0.0f);
		selection_ctx.editor_rotation_deg = glm::vec3(0.0f);
	}
	return true;
}

bool syncSelectedChameleonMaterial(const Scene* scene, float voice_loudness) {
	const int preset_index = selectedChameleonPresetIndex(scene);
	if (preset_index < 0) {
		return false;
	}
	selection_ctx.voice_parameter = VoiceDrivenParameter::ChameleonMaterial;
	if (selection_ctx.material_edit_mode != MaterialEditMode::Voice) {
		return false;
	}

	const ChameleonMaterialTransition& transition =
	    kChameleonMaterialTransitions[static_cast<size_t>(preset_index)];
	const float level = std::clamp(voice_loudness, 0.0f, 1.0f);

	const glm::vec3 target_color = interpolateChameleonColor(transition, level);
	const float target_metalness = transition.start_metalness +
	                               level * (transition.end_metalness - transition.start_metalness);
	const float target_roughness = transition.start_roughness +
	                               level * (transition.end_roughness - transition.start_roughness);
	const float target_transmission =
	    transition.start_transmission +
	    level * (transition.end_transmission - transition.start_transmission);
	const float target_texture_weight = 1.0f - level;

	const GPUMaterial& current = selection_ctx.editor_material;
	const bool changed = glm::length(glm::vec3(current.base_color) - target_color) > 1.0e-4f ||
	                     std::abs(current.base_metalness - target_metalness) > 1.0e-4f ||
	                     std::abs(current.specular_roughness - target_roughness) > 1.0e-4f ||
	                     std::abs(current.transmission_weight - target_transmission) > 1.0e-4f ||
	                     std::abs(current.texture_weight - target_texture_weight) > 1.0e-4f;
	if (!changed) {
		return false;
	}

	// Texture indices stay intact so the fade is reversible, but their shader influence
	// reaches zero at the end state.
	selection_ctx.editor_material.base_color =
	    glm::vec4(target_color, selection_ctx.editor_material.base_color.a);
	selection_ctx.editor_material.base_metalness      = target_metalness;
	selection_ctx.editor_material.specular_roughness  = target_roughness;
	selection_ctx.editor_material.transmission_weight = target_transmission;
	selection_ctx.editor_material.texture_weight      = target_texture_weight;
	return true;
}

static void updateMaterialBufferSlot(VmaAllocator allocator, void* material_mapped,
                                     uint32_t material_count, VmaAllocation material_alloc,
                                     int material_index, const GPUMaterial& material) {
	if (material_mapped == nullptr || material_index < 0 ||
	    material_index >= static_cast<int>(material_count)) {
		return;
	}

	auto* gpu_materials           = reinterpret_cast<GPUMaterial*>(material_mapped);
	gpu_materials[material_index] = material;
	vmaFlushAllocation(allocator, material_alloc,
	                   static_cast<VkDeviceSize>(material_index) * sizeof(GPUMaterial),
	                   sizeof(GPUMaterial));
}

void applySelectedMaterialEditor(Scene* scene, VmaAllocator allocator, void* material_mapped,
                                 uint32_t material_count, VmaAllocation material_alloc) {
	if (scene == nullptr || selection_ctx.selected_mesh_index < 0 ||
	    selection_ctx.selected_mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		return;
	}

	auto& mesh = scene->m_meshes[selection_ctx.selected_mesh_index];
	if (mesh == nullptr || mesh->m_material == nullptr) {
		return;
	}

	mesh->m_material->m_gpu = selection_ctx.editor_material;
	updateMaterialBufferSlot(allocator, material_mapped, material_count, material_alloc,
	                         selection_ctx.selected_mesh_index, selection_ctx.editor_material);
}

SelectionPanelResult drawSelectionPanel(const Scene* scene, float voice_loudness, bool* is_open) {
	SelectionPanelResult result{};
	const int            chameleon_preset_index = selectedChameleonPresetIndex(scene);
	const glm::vec3      voice_color =
	    chameleon_preset_index >= 0 ? glm::vec3(0.0f) : rainbowColorFromLoudness(voice_loudness);
	const float voice_value = voiceDrivenScalarValue(selection_ctx.voice_parameter, voice_loudness);

	if (is_open != nullptr && !*is_open) {
		return result;
	}

	ImGui::SetNextWindowPos(ImVec2(470.0f, 24.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360.0f, 320.0f), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Object Inspector", is_open)) {
		ImGui::End();
		return result;
	}

	ImGui::TextUnformatted("Click the render view to select a mesh.");

	int         edit_mode  = static_cast<int>(selection_ctx.material_edit_mode);
	const char* edit_items = "GUI\0Voice\0";
	if (ImGui::Combo("Inspector input", &edit_mode, edit_items)) {
		selection_ctx.material_edit_mode = static_cast<MaterialEditMode>(edit_mode);
	}
	if (selection_ctx.material_edit_mode == MaterialEditMode::Voice) {
		if (chameleon_preset_index >= 0) {
			int chameleon_target = 0;
			ImGui::BeginDisabled();
			ImGui::Combo("Voice target", &chameleon_target,
			             "Chameleon material (all fields + texture fade)\0");
			ImGui::EndDisabled();
		} else {
			int         voice_parameter = static_cast<int>(selection_ctx.voice_parameter);
			const char* voice_items =
			    "Base tint\0Emission color\0Metalness\0Roughness\0Transmission\0IOR\0Emission "
			    "intensity\0Object scale\0Translate X\0Translate Y\0Translate Z\0Rotate X\0Rotate "
			    "Y\0Rotate Z\0";
			if (ImGui::Combo("Voice target", &voice_parameter, voice_items)) {
				selection_ctx.voice_parameter = static_cast<VoiceDrivenParameter>(voice_parameter);
			}
		}
	}

	int         debug_view_mode  = static_cast<int>(selection_ctx.debug_view_mode);
	const char* debug_view_items = "HiPR\0Obj ID\0HiPR Vis\0Naive\0";
	if (ImGui::Combo("Renderer view", &debug_view_mode, debug_view_items)) {
		selection_ctx.debug_view_mode = static_cast<RenderDebugViewMode>(debug_view_mode);
	}

	if (ImGui::CollapsingHeader("Path Tracing", ImGuiTreeNodeFlags_DefaultOpen)) {
		uint64_t       spp       = selection_ctx.path_tracing.spp;
		const uint64_t unit_step = 1ull;
		if (ImGui::InputScalar("SPP", ImGuiDataType_U64, &spp, &unit_step, &unit_step, "%llu")) {
			selection_ctx.path_tracing.spp =
			    static_cast<uint32_t>(std::clamp<uint64_t>(spp, 1ull, 1024ull));
			result.path_tracing_settings_changed = true;
		}

		uint64_t max_bounces = selection_ctx.path_tracing.max_bounces;
		if (ImGui::InputScalar("Max bounces", ImGuiDataType_U64, &max_bounces, &unit_step,
		                       &unit_step, "%llu")) {
			selection_ctx.path_tracing.max_bounces =
			    static_cast<uint32_t>(std::clamp<uint64_t>(max_bounces, 1ull, 1024ull));
			result.path_tracing_settings_changed = true;
		}

		uint64_t water_spp_override = selection_ctx.path_tracing.hipr_water_spp_override;
		if (ImGui::InputScalar("HiPR water SPP override (0=off)", ImGuiDataType_U64,
		                       &water_spp_override, &unit_step, &unit_step, "%llu")) {
			selection_ctx.path_tracing.hipr_water_spp_override =
			    static_cast<uint32_t>(std::clamp<uint64_t>(water_spp_override, 0ull, 1024ull));
			result.path_tracing_settings_changed = true;
		}
	}

	if (ImGui::CollapsingHeader("HiPR Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
		uint64_t rank_count = selection_ctx.hipr_debug.rank_count;
		if (ImGui::InputScalar("Top-K objects", ImGuiDataType_U64, &rank_count, nullptr, nullptr,
		                       "%llu")) {
			const uint32_t previous             = selection_ctx.hipr_debug.rank_count;
			selection_ctx.hipr_debug.rank_count = sanitizeTopKValue(rank_count);
			result.hipr_settings_changed |= (selection_ctx.hipr_debug.rank_count != previous);
		}

		uint64_t       frames_per_object = selection_ctx.hipr_debug.frames_per_object;
		const uint64_t frame_step        = 10ull;
		if (ImGui::InputScalar("Frames per object", ImGuiDataType_U64, &frames_per_object,
		                       &frame_step, &frame_step, "%llu")) {
			const uint32_t previous = selection_ctx.hipr_debug.frames_per_object;
			selection_ctx.hipr_debug.frames_per_object =
			    sanitizeTenStepFrameValue(frames_per_object);
			result.hipr_settings_changed |=
			    (selection_ctx.hipr_debug.frames_per_object != previous);
		}

		if (selection_ctx.debug_view_mode == RenderDebugViewMode::HiPRVis) {
			ImGui::SeparatorText("HiPR Vis");
			ImGui::Checkbox("Influence tint", &selection_ctx.hipr_debug.vis_enable_influence_tint);
			ImGui::BeginDisabled(!selection_ctx.hipr_debug.vis_enable_influence_tint);
			ImGui::Checkbox("Heatmap tint", &selection_ctx.hipr_debug.vis_rainbow_tint);
			ImGui::SliderFloat("Tint strength", &selection_ctx.hipr_debug.vis_tint_strength, 0.0f,
			                   1.0f, "%.2f");
			ImGui::EndDisabled();
		}
	}

	if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Skybox", &selection_ctx.lighting.skybox_enabled);
		ImGui::Checkbox("Directional light", &selection_ctx.lighting.directional_light_enabled);
		ImGui::BeginDisabled(!selection_ctx.lighting.directional_light_enabled);
		ImGui::SliderFloat("Elevation", &selection_ctx.lighting.sun_elevation_deg, -90.0f, 90.0f,
		                   "%.1f deg");
		ImGui::SliderFloat("Azimuth", &selection_ctx.lighting.sun_azimuth_deg, -180.0f, 180.0f,
		                   "%.1f deg");
		ImGui::SliderFloat("Sun intensity", &selection_ctx.lighting.sun_intensity, 0.0f, 50.0f,
		                   "%.1f");
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat("FOV", &selection_ctx.camera.fov_deg, 20.0f, 120.0f, "%.1f deg");
		ImGui::Text("Selected mesh: %s",
		            meshDisplayName(scene, selection_ctx.selected_mesh_index).c_str());
	}

	const bool has_selection =
	    scene != nullptr && selection_ctx.selected_mesh_index >= 0 &&
	    selection_ctx.selected_mesh_index < static_cast<int>(scene->m_meshes.size());

	if (has_selection) {
		if (ImGui::Button("Clear selection")) {
			result.selection_changed = selectMesh(scene, -1);
		}
	} else {
		ImGui::TextUnformatted("No mesh selected.");
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Selection outline");
	int outline_width = static_cast<int>(selection_ctx.outline_width);
	if (ImGui::SliderInt("Outline width", &outline_width, 1, 4)) {
		selection_ctx.outline_width = static_cast<uint32_t>(outline_width);
	}
	ImGui::ColorEdit4("Outline color", glm::value_ptr(selection_ctx.outline_color),
	                  ImGuiColorEditFlags_AlphaBar);

	if (!has_selection) {
		if (selection_ctx.material_edit_mode == MaterialEditMode::Voice) {
			ImGui::Separator();
			ImGui::Text("Voice target: %s",
			            voiceDrivenParameterLabel(selection_ctx.voice_parameter));
			ImGui::Text("Voice loudness: %.2f", voice_loudness);
			if (voiceParameterUsesRainbowColor(selection_ctx.voice_parameter)) {
				ImGui::ColorButton("Voice preview",
				                   ImVec4(voice_color.r, voice_color.g, voice_color.b, 1.0f),
				                   ImGuiColorEditFlags_NoTooltip, ImVec2(52.0f, 20.0f));
				ImGui::TextWrapped(
				    "This target uses the rainbow loudness map. Select a mesh to apply it.");
			} else {
				ImGui::Text("Mapped value: %.2f", voice_value);
				ImGui::TextWrapped("Select a mesh to apply the current voice-driven value.");
			}
		}
		ImGui::End();
		return result;
	}

	ImGui::Separator();

	const bool gui_mode_enabled = selection_ctx.material_edit_mode == MaterialEditMode::Gui;
	if (chameleon_preset_index >= 0 && !gui_mode_enabled) {
		const ChameleonMaterialTransition& transition =
		    kChameleonMaterialTransitions[static_cast<size_t>(chameleon_preset_index)];
		const float     level           = std::clamp(voice_loudness, 0.0f, 1.0f);
		const glm::vec3 chameleon_color = interpolateChameleonColor(transition, level);
		ImGui::Text("Chameleon preset: %d", chameleon_preset_index + 1);
		ImGui::Text("Voice loudness: %.2f", voice_loudness);
		ImGui::ColorButton("Chameleon color preview",
		                   ImVec4(chameleon_color.r, chameleon_color.g, chameleon_color.b, 1.0f),
		                   ImGuiColorEditFlags_NoTooltip, ImVec2(52.0f, 20.0f));
		ImGui::ProgressBar(level, ImVec2(-1.0f, 0.0f));
		ImGui::TextWrapped(
		    "Base color, metalness, roughness, transmission, and texture influence are blended "
		    "by voice loudness while this chameleon is selected.");
	} else if (!gui_mode_enabled) {
		const VoiceDrivenSyncResult voice_sync = syncVoiceDrivenSelectionParameter(voice_loudness);
		result.material_changed |= voice_sync.material_changed;
		result.transform_changed |= voice_sync.transform_changed;

		ImGui::Text("Voice target: %s", voiceDrivenParameterLabel(selection_ctx.voice_parameter));
		ImGui::Text("Voice loudness: %.2f", voice_loudness);
		if (voiceParameterUsesRainbowColor(selection_ctx.voice_parameter)) {
			ImGui::ColorButton("Voice preview",
			                   ImVec4(voice_color.r, voice_color.g, voice_color.b, 1.0f),
			                   ImGuiColorEditFlags_NoTooltip, ImVec2(52.0f, 20.0f));
			ImGui::TextWrapped("This parameter is currently driven by the rainbow loudness map.");
		} else {
			ImGui::Text("Mapped value: %.2f", voice_value);
			ImGui::TextWrapped("Only the selected voice target changes with loudness.");
		}
	}

	ImGui::SeparatorText("Transform");
	ImGui::BeginDisabled(!gui_mode_enabled);
	result.transform_changed |=
	    ImGui::SliderFloat("Scale", &selection_ctx.editor_scale, 0.10f, 3.00f, "%.2f");
	result.transform_changed |=
	    ImGui::SliderFloat3("Translation (world)", glm::value_ptr(selection_ctx.editor_translation),
	                        kEditorTranslationMin, kEditorTranslationMax, "%.2f");
	result.transform_changed |=
	    ImGui::SliderFloat3("Rotation", glm::value_ptr(selection_ctx.editor_rotation_deg),
	                        kEditorRotationMinDeg, kEditorRotationMaxDeg, "%.1f deg");
	if (ImGui::Button("Reset transform")) {
		selection_ctx.editor_scale        = 1.0f;
		selection_ctx.editor_translation  = glm::vec3(0.0f);
		selection_ctx.editor_rotation_deg = glm::vec3(0.0f);
		result.transform_changed          = true;
	}
	ImGui::EndDisabled();

	ImGui::BeginDisabled(!gui_mode_enabled);

	auto record_item_edit_state = [&result]() {
		result.material_edit_active |= ImGui::IsItemActive();
		result.material_edit_just_finished |= ImGui::IsItemDeactivatedAfterEdit();
	};

	auto mark_changed = [&](bool changed) {
		result.material_changed |= changed;
		record_item_edit_state();
	};

	auto slider_float = [&](const char* label, float* value, float min_value, float max_value,
	                        const char* fmt) {
		mark_changed(ImGui::SliderFloat(label, value, min_value, max_value, fmt));
	};

	auto color_edit3 = [&](const char* label, glm::vec4& value) {
		mark_changed(ImGui::ColorEdit3(label, glm::value_ptr(value)));
	};

	auto color_edit4 = [&](const char* label, glm::vec4& value, ImGuiColorEditFlags flags = 0) {
		mark_changed(ImGui::ColorEdit4(label, glm::value_ptr(value), flags));
	};

	ImGui::SeparatorText("Base");
	color_edit4("Base color + weight##base", selection_ctx.editor_material.base_color,
	            ImGuiColorEditFlags_AlphaBar);
	slider_float("Base metalness##base", &selection_ctx.editor_material.base_metalness, 0.0f, 1.0f,
	             "%.3f");
	slider_float("Base diffuse roughness##base",
	             &selection_ctx.editor_material.base_diffuse_roughness, 0.0f, 1.0f, "%.3f");

	ImGui::SeparatorText("Specular");
	color_edit3("Specular color##spec", selection_ctx.editor_material.specular_color);
	slider_float("Specular weight##spec", &selection_ctx.editor_material.specular_color.a, 0.0f,
	             1.0f, "%.3f");
	slider_float("Specular roughness##spec", &selection_ctx.editor_material.specular_roughness,
	             0.02f, 1.0f, "%.3f");
	slider_float("Specular IOR##spec", &selection_ctx.editor_material.specular_ior, 1.0f, 2.5f,
	             "%.3f");
	slider_float("Specular anisotropy##spec", &selection_ctx.editor_material.specular_anisotropy,
	             -1.0f, 1.0f, "%.3f");

	ImGui::SeparatorText("Transmission");
	slider_float("Transmission weight##tr", &selection_ctx.editor_material.transmission_weight,
	             0.0f, 1.0f, "%.3f");
	slider_float("Opacity##tr", &selection_ctx.editor_material.geometry_opacity, 0.0f, 1.0f,
	             "%.3f");
	color_edit3("Transmission color##tr", selection_ctx.editor_material.transmission_color);
	slider_float("Transmission depth##tr", &selection_ctx.editor_material.transmission_depth, 0.0f,
	             10.0f, "%.3f");
	color_edit3("Transmission scatter##tr", selection_ctx.editor_material.transmission_scatter);
	slider_float("Scatter anisotropy##tr",
	             &selection_ctx.editor_material.transmission_scatter_anisotropy, -1.0f, 1.0f,
	             "%.3f");
	slider_float("Dispersion scale##tr",
	             &selection_ctx.editor_material.transmission_dispersion_scale, 0.0f, 1.0f, "%.3f");
	slider_float("Abbe number##tr",
	             &selection_ctx.editor_material.transmission_dispersion_abbe_number, 0.0f, 200.0f,
	             "%.1f");
	{
		bool       thin_walled = selection_ctx.editor_material.geometry_thin_walled > 0.5f;
		const bool changed     = ImGui::Checkbox("Thin walled##tr", &thin_walled);
		if (changed) {
			selection_ctx.editor_material.geometry_thin_walled = thin_walled ? 1.0f : 0.0f;
		}
		mark_changed(changed);
	}

	ImGui::SeparatorText("Coat");
	slider_float("Coat weight##coat", &selection_ctx.editor_material.coat_weight, 0.0f, 1.0f,
	             "%.3f");
	color_edit3("Coat color##coat", selection_ctx.editor_material.coat_color);
	slider_float("Coat roughness##coat", &selection_ctx.editor_material.coat_roughness, 0.0f, 1.0f,
	             "%.3f");
	slider_float("Coat IOR##coat", &selection_ctx.editor_material.coat_ior, 1.0f, 2.5f, "%.3f");

	ImGui::SeparatorText("Fuzz");
	slider_float("Fuzz weight##fuzz", &selection_ctx.editor_material.fuzz_weight, 0.0f, 1.0f,
	             "%.3f");
	color_edit3("Fuzz color##fuzz", selection_ctx.editor_material.fuzz_color);
	slider_float("Fuzz roughness##fuzz", &selection_ctx.editor_material.fuzz_roughness, 0.0f, 1.0f,
	             "%.3f");

	ImGui::SeparatorText("Thin Film");
	slider_float("Thin film weight##film", &selection_ctx.editor_material.thin_film_weight, 0.0f,
	             1.0f, "%.3f");
	slider_float("Thin film IOR##film", &selection_ctx.editor_material.thin_film_ior, 1.0f, 2.5f,
	             "%.3f");
	slider_float("Thin film thickness##film", &selection_ctx.editor_material.thin_film_thickness,
	             0.0f, 3000.0f, "%.1f");

	ImGui::SeparatorText("Emission");
	color_edit3("Emission color##emiss", selection_ctx.editor_material.emission_color);
	slider_float("Emission intensity##emiss", &selection_ctx.editor_material.emission_luminance,
	             0.0f, 100.0f, "%.3f");

	ImGui::EndDisabled();

	ImGui::End();
	return result;
}

}        // namespace ui
