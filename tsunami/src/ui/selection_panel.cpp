#include "tsunami/ui/selection_panel.h"

#include "imgui.h"

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace ui {

SelectionContext::SelectionContext() {
	editor_material = Material{}.pack();
}

SelectionContext selection_ctx{};

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

static const ObjectIdEntry* objectIdEntryForId(int object_id) {
	if (object_id < 0 || object_id >= static_cast<int>(selection_ctx.object_id_map.size())) {
		return nullptr;
	}
	return &selection_ctx.object_id_map[object_id];
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

bool selectMesh(const Scene* scene, int mesh_index) {
	const int max_mesh_index =
	    (scene != nullptr) ? static_cast<int>(scene->m_meshes.size()) - 1 : -1;
	const int clamped_index = (mesh_index >= 0 && mesh_index <= max_mesh_index) ? mesh_index : -1;
	if (selection_ctx.selected_mesh_index == clamped_index) {
		return false;
	}

	selection_ctx.selected_mesh_index = clamped_index;
	refreshSelectedMaterialEditor(scene);
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

SelectionPanelResult drawSelectionPanel(const Scene* scene) {
	SelectionPanelResult result{};

	ImGui::SetNextWindowPos(ImVec2(470.0f, 24.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360.0f, 320.0f), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Object Inspector")) {
		ImGui::End();
		return result;
	}

	ImGui::TextUnformatted("Click the render view to select a mesh.");

	int         edit_mode  = static_cast<int>(selection_ctx.material_edit_mode);
	const char* edit_items = "GUI\0Voice\0";
	if (ImGui::Combo("Material input", &edit_mode, edit_items)) {
		selection_ctx.material_edit_mode = static_cast<MaterialEditMode>(edit_mode);
	}

	int         debug_view_mode  = static_cast<int>(selection_ctx.debug_view_mode);
	const char* debug_view_items = "HiPR\0Obj ID\0HiPR Vis\0Naive\0";
	if (ImGui::Combo("Renderer view", &debug_view_mode, debug_view_items)) {
		selection_ctx.debug_view_mode = static_cast<RenderDebugViewMode>(debug_view_mode);
	}

	if (ImGui::CollapsingHeader("HiPR Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
		int rank_count = static_cast<int>(selection_ctx.hipr_debug.rank_count);
		if (ImGui::SliderInt("Top-K objects", &rank_count, 1, 16)) {
			selection_ctx.hipr_debug.rank_count = static_cast<uint32_t>(rank_count);
		}

		int frames_per_object = static_cast<int>(selection_ctx.hipr_debug.frames_per_object);
		if (ImGui::SliderInt("Frames per object", &frames_per_object, 1, 2000)) {
			selection_ctx.hipr_debug.frames_per_object = static_cast<uint32_t>(frames_per_object);
		}

		int update_period = static_cast<int>(selection_ctx.hipr_debug.update_period_frames);
		if (ImGui::SliderInt("Resort every N frames", &update_period, 1, 120)) {
			selection_ctx.hipr_debug.update_period_frames = static_cast<uint32_t>(update_period);
		}

		ImGui::Checkbox("Incremental stable sorting",
		                &selection_ctx.hipr_debug.incremental_sorting);
		ImGui::Checkbox("Full resort on material change",
		                &selection_ctx.hipr_debug.full_resort_on_material_change);

		ImGui::BeginDisabled(!selection_ctx.hipr_debug.incremental_sorting);
		ImGui::SliderFloat("Score blend", &selection_ctx.hipr_debug.score_blend, 0.05f, 1.0f,
		                   "%.2f");
		ImGui::EndDisabled();

		ImGui::SeparatorText("HiPR Vis");
		ImGui::Checkbox("Influence tint", &selection_ctx.hipr_debug.vis_enable_influence_tint);
		ImGui::BeginDisabled(!selection_ctx.hipr_debug.vis_enable_influence_tint);
		ImGui::Checkbox("Heatmap tint", &selection_ctx.hipr_debug.vis_rainbow_tint);
		ImGui::SliderFloat("Tint strength", &selection_ctx.hipr_debug.vis_tint_strength, 0.0f, 1.0f,
		                   "%.2f");
		ImGui::EndDisabled();
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
	}

	ImGui::Text("Scene objects: %d",
	            scene != nullptr ? static_cast<int>(scene->m_meshes.size()) : 0);
	ImGui::Text("Object IDs: %d", static_cast<int>(selection_ctx.object_id_map.size()));
	ImGui::Text("Selected mesh: %s",
	            meshDisplayName(scene, selection_ctx.selected_mesh_index).c_str());

	const bool has_selection =
	    scene != nullptr && selection_ctx.selected_mesh_index >= 0 &&
	    selection_ctx.selected_mesh_index < static_cast<int>(scene->m_meshes.size());

	if (has_selection) {
		const ObjectIdEntry* selected_entry = objectIdEntryForId(selection_ctx.selected_mesh_index);
		ImGui::Text("Object ID: %d", selected_entry != nullptr ? selected_entry->object_id : -1);
		ImGui::Text("Mesh index: %d", selection_ctx.selected_mesh_index);
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

	if (ImGui::CollapsingHeader("Object ID Map")) {
		for (const ObjectIdEntry& entry : selection_ctx.object_id_map) {
			ImGui::Text("ID %d -> %s", entry.object_id, entry.display_name.c_str());
		}
	}

	if (!has_selection) {
		if (selection_ctx.material_edit_mode == MaterialEditMode::Voice) {
			ImGui::Separator();
			ImGui::TextWrapped(
			    "Voice mode is selected, but there is not yet a speech-to-text command layer for "
			    "material edits in this project.");
		}
		ImGui::End();
		return result;
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Material");
	ImGui::TextWrapped(
	    "Texture-backed meshes use these controls as live multipliers and overrides.");

	const bool gui_mode_enabled = selection_ctx.material_edit_mode == MaterialEditMode::Gui;
	ImGui::BeginDisabled(!gui_mode_enabled);

	auto record_item_edit_state = [&result]() {
		result.material_edit_active |= ImGui::IsItemActive();
		result.material_edit_just_finished |= ImGui::IsItemDeactivatedAfterEdit();
	};

	if (ImGui::ColorEdit3("Base tint", glm::value_ptr(selection_ctx.editor_material.base_color))) {
		result.material_changed = true;
	}
	record_item_edit_state();

	if (ImGui::SliderFloat("Opacity", &selection_ctx.editor_material.geometry_opacity, 0.0f, 1.0f,
	                       "%.2f")) {
		result.material_changed = true;
	}
	record_item_edit_state();

	if (ImGui::SliderFloat("Metalness", &selection_ctx.editor_material.base_metalness, 0.0f, 1.0f,
	                       "%.2f")) {
		result.material_changed = true;
	}
	record_item_edit_state();

	if (ImGui::SliderFloat("Roughness", &selection_ctx.editor_material.specular_roughness, 0.02f,
	                       1.0f, "%.2f")) {
		result.material_changed = true;
	}
	record_item_edit_state();

	if (ImGui::SliderFloat("Transmission", &selection_ctx.editor_material.transmission_weight, 0.0f,
	                       1.0f, "%.2f")) {
		result.material_changed = true;
	}
	record_item_edit_state();

	if (ImGui::SliderFloat("IOR", &selection_ctx.editor_material.specular_ior, 1.0f, 2.5f,
	                       "%.2f")) {
		result.material_changed = true;
	}
	record_item_edit_state();

	if (ImGui::ColorEdit3("Emission color",
	                      glm::value_ptr(selection_ctx.editor_material.emission_color))) {
		result.material_changed = true;
	}
	record_item_edit_state();

	if (ImGui::SliderFloat("Emission intensity", &selection_ctx.editor_material.emission_luminance,
	                       0.0f, 20.0f, "%.2f")) {
		result.material_changed = true;
	}
	record_item_edit_state();

	bool thin_walled = selection_ctx.editor_material.geometry_thin_walled > 0.5f;
	if (ImGui::Checkbox("Thin walled", &thin_walled)) {
		selection_ctx.editor_material.geometry_thin_walled = thin_walled ? 1.0f : 0.0f;
		result.material_changed                            = true;
	}
	record_item_edit_state();

	ImGui::EndDisabled();

	if (!gui_mode_enabled) {
		ImGui::TextWrapped(
		    "Voice mode is selected, but there is not yet a speech-to-text command layer for "
		    "material edits in this project.");
	}

	ImGui::End();
	return result;
}

}        // namespace ui
