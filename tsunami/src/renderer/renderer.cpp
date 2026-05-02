#include "tsunami/renderer/renderer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <glm/glm.hpp>

#include "imgui.h"

#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/camera/camera.h"
#include "tsunami/camera/fly_camera.h"
#include "tsunami/core/window.h"
#include "tsunami/scene/scene.h"
#include "tsunami/simulation/water_surface_simulation.h"
#include "tsunami/ui/audience_control_panel.h"
#include "tsunami/ui/audience_overlay.h"
#include "tsunami/ui/selection_panel.h"
#include "tsunami/vulkan/vk_scene.h"
#include "tsunami/vulkan/vulkan.h"

namespace {

constexpr uint32_t kMaxHiPRTopK = 16u;

void applyOverlayLevel(ui::AudienceControlPanelState& controls, float audio_level) {
	controls.overlay.volume_level = std::clamp(audio_level, 0.0f, 1.0f);
	controls.overlay.selected_index =
	    ui::quantizeSelection(controls.overlay.volume_level, controls.overlay.selection_count);
}

uint32_t sanitizeHiPRFramesPerObject(uint32_t value) {
	if (value <= 1u) {
		return 1u;
	}
	if (value >= 100u) {
		return 100u;
	}

	uint32_t snapped = ((value + 5u) / 10u) * 10u;
	snapped          = std::max(snapped, 10u);
	snapped          = std::min(snapped, 100u);
	return snapped;
}

struct CpuRay {
	glm::vec3 origin{};
	glm::vec3 direction{};
};

glm::vec2 cursorPositionInFramebuffer(GLFWwindow* window, uint32_t framebuffer_width,
                                      uint32_t framebuffer_height) {
	double cursor_x = 0.0;
	double cursor_y = 0.0;
	glfwGetCursorPos(window, &cursor_x, &cursor_y);

	int window_width  = 1;
	int window_height = 1;
	glfwGetWindowSize(window, &window_width, &window_height);

	const float scale_x = window_width > 0 ? static_cast<float>(framebuffer_width) /
	                                             static_cast<float>(window_width) :
	                                         1.0f;
	const float scale_y = window_height > 0 ? static_cast<float>(framebuffer_height) /
	                                              static_cast<float>(window_height) :
	                                          1.0f;

	return glm::vec2(static_cast<float>(cursor_x) * scale_x,
	                 static_cast<float>(cursor_y) * scale_y);
}

CpuRay buildPickRay(const GPUCamera& camera, uint32_t framebuffer_width,
                    uint32_t framebuffer_height, const glm::vec2& cursor_position) {
	const glm::vec3 origin = glm::vec3(camera.position);
	const glm::vec3 target = glm::vec3(camera.target);
	const glm::vec3 up     = glm::normalize(glm::vec3(camera.up));
	const float     fov    = camera.fov_near_far.x;
	const float     aspect = static_cast<float>(framebuffer_width) /
	                     static_cast<float>(std::max(framebuffer_height, 1u));

	const float half_height = std::tan(glm::radians(fov) * 0.5f);
	const float half_width  = aspect * half_height;

	const glm::vec3 forward = glm::normalize(target - origin);
	const glm::vec3 right   = glm::normalize(glm::cross(forward, up));
	const glm::vec3 up_axis = glm::cross(right, forward);

	const glm::vec2 clamped_cursor = glm::clamp(
	    cursor_position, glm::vec2(0.0f),
	    glm::vec2(static_cast<float>(framebuffer_width), static_cast<float>(framebuffer_height)));
	const glm::vec2 uv =
	    glm::vec2(clamped_cursor.x / static_cast<float>(std::max(framebuffer_width, 1u)),
	              clamped_cursor.y / static_cast<float>(std::max(framebuffer_height, 1u)));

	const float u = (2.0f * uv.x - 1.0f) * half_width;
	const float v = (1.0f - 2.0f * uv.y) * half_height;

	return CpuRay{origin, glm::normalize(forward + u * right + v * up_axis)};
}

bool intersectRayAabb(const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& min,
                      const glm::vec3& max, float& out_t_min, float& out_t_max) {
	float t_min = 0.0f;
	float t_max = std::numeric_limits<float>::infinity();

	for (int axis = 0; axis < 3; ++axis) {
		const float dir = direction[axis];
		const float ori = origin[axis];

		if (std::abs(dir) < 1.0e-8f) {
			if (ori < min[axis] || ori > max[axis]) {
				return false;
			}
			continue;
		}

		float t0 = (min[axis] - ori) / dir;
		float t1 = (max[axis] - ori) / dir;
		if (t0 > t1) {
			std::swap(t0, t1);
		}

		t_min = std::max(t_min, t0);
		t_max = std::min(t_max, t1);
		if (t_min > t_max) {
			return false;
		}
	}

	out_t_min = t_min;
	out_t_max = t_max;
	return t_max >= 0.0f;
}

bool intersectRayTriangle(const CpuRay& ray, const glm::vec3& v0, const glm::vec3& v1,
                          const glm::vec3& v2, float& out_t) {
	constexpr float kEpsilon = 1.0e-7f;

	const glm::vec3 edge1 = v1 - v0;
	const glm::vec3 edge2 = v2 - v0;
	const glm::vec3 pvec  = glm::cross(ray.direction, edge2);
	const float     det   = glm::dot(edge1, pvec);

	if (std::abs(det) < kEpsilon) {
		return false;
	}

	const float     inv_det = 1.0f / det;
	const glm::vec3 tvec    = ray.origin - v0;
	const float     u       = glm::dot(tvec, pvec) * inv_det;
	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	const glm::vec3 qvec = glm::cross(tvec, edge1);
	const float     v    = glm::dot(ray.direction, qvec) * inv_det;
	if (v < 0.0f || (u + v) > 1.0f) {
		return false;
	}

	const float t = glm::dot(edge2, qvec) * inv_det;
	if (t <= kEpsilon) {
		return false;
	}

	out_t = t;
	return true;
}

int pickMeshAtCursor(const Scene* scene, GLFWwindow* window, const GPUCamera& camera,
                     uint32_t framebuffer_width, uint32_t framebuffer_height) {
	if (scene == nullptr || window == nullptr || framebuffer_width == 0 ||
	    framebuffer_height == 0) {
		return -1;
	}

	const glm::vec2 cursor =
	    cursorPositionInFramebuffer(window, framebuffer_width, framebuffer_height);
	if (cursor.x < 0.0f || cursor.y < 0.0f || cursor.x >= static_cast<float>(framebuffer_width) ||
	    cursor.y >= static_cast<float>(framebuffer_height)) {
		return -1;
	}

	const CpuRay world_ray    = buildPickRay(camera, framebuffer_width, framebuffer_height, cursor);
	float        best_t       = std::numeric_limits<float>::infinity();
	int          best_mesh_id = -1;

	for (int mesh_index = 0; mesh_index < static_cast<int>(scene->m_meshes.size()); ++mesh_index) {
		const auto& mesh = scene->m_meshes[mesh_index];
		if (mesh == nullptr || mesh->gpuIndices.size() < 3 || mesh->gpuVertices.empty()) {
			continue;
		}

		const glm::mat4& inverse_transform = mesh->m_transform.m_inverseTransform;
		const CpuRay     local_ray{
            glm::vec3(inverse_transform * glm::vec4(world_ray.origin, 1.0f)),
            glm::vec3(inverse_transform * glm::vec4(world_ray.direction, 0.0f)),
        };

		if (glm::dot(local_ray.direction, local_ray.direction) < 1.0e-12f) {
			continue;
		}

		float aabb_t_min = 0.0f;
		float aabb_t_max = 0.0f;
		if (!intersectRayAabb(local_ray.origin, local_ray.direction, mesh->m_local_bounds_min,
		                      mesh->m_local_bounds_max, aabb_t_min, aabb_t_max) ||
		    aabb_t_min > best_t) {
			continue;
		}

		for (size_t index = 0; index + 2 < mesh->gpuIndices.size(); index += 3) {
			const glm::vec3& v0 = mesh->gpuVertices[mesh->gpuIndices[index + 0]].position;
			const glm::vec3& v1 = mesh->gpuVertices[mesh->gpuIndices[index + 1]].position;
			const glm::vec3& v2 = mesh->gpuVertices[mesh->gpuIndices[index + 2]].position;

			float hit_t = 0.0f;
			if (intersectRayTriangle(local_ray, v0, v1, v2, hit_t) && hit_t < best_t) {
				best_t       = hit_t;
				best_mesh_id = mesh_index;
			}
		}
	}

	return best_mesh_id;
}

}        // namespace

namespace renderer {

class Renderer::Impl {
  public:
	Impl(core::Window& window, const std::string& scene_argument) : m_window(&window) {
		m_scene.m_camera = Camera(glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f),
		                          glm::vec3(0.0f, 1.0f, 0.0f), 60.0f, 0.1f, 10000.0f);
		m_scene.load_gltf(resolveScenePathOrThrow(scene_argument));
		ui::rebuildObjectIdMap(m_selection, &m_scene);
		m_selection.camera.fov_deg = m_scene.m_camera.m_fov;

		m_backend = std::make_unique<vulkan::Runtime>(*m_window, m_scene);
		recreateWaterSurface();

		m_fly_camera = std::make_unique<FlyCamera>(
		    m_scene.m_camera.m_position, m_scene.m_camera.m_target, m_scene.m_camera.m_fov, 0.5f,
		    0.0005f, m_scene.m_camera.m_nearClip, m_scene.m_camera.m_farClip,
		    m_scene.m_camera.m_up);
		const GPUCamera initial_camera = m_fly_camera->pack();
		syncSceneCamera(initial_camera);
		m_backend->updateCamera(initial_camera);
	}

	~Impl() {
		if (m_backend != nullptr) {
			m_backend->waitIdle();
		}
		m_water_surface.reset();
		m_backend.reset();
	}

	void tick(float time_seconds, float delta_time,
	          const audio::ReactiveAudioInputFrame& audio_frame) {
		const uint32_t framebuffer_width  = m_window->width();
		const uint32_t framebuffer_height = m_window->height();

		vulkan::RuntimeFrameOutput frame_output = m_backend->beginFrame();
		if (frame_output.surface_resized) {
			recreateWaterSurface();
		}
		m_diagnostics.render = frame_output.render;

		handleHotkeys();
		updateAudio(audio_frame, delta_time);

		bool controls_changed = false;
		if (m_show_all_gui && m_show_control_panel) {
			controls_changed =
			    ui::drawAudienceControlPanel(&m_show_control_panel, m_controls, m_diagnostics);
		}

		ui::SelectionPanelResult selection_panel_result{};
		if (m_show_all_gui && m_show_selection_panel) {
			selection_panel_result = ui::drawSelectionPanel(
			    m_selection, &m_scene, m_selection_voice_loudness, &m_show_selection_panel);
		}
		sanitizeSelectionState(selection_panel_result);

		if (controls_changed) {
			updateAudio(audio_frame, delta_time);
		}

		if (selection_panel_result.material_changed) {
			ui::applySelectedMaterialEditor(m_selection, &m_scene);
			m_backend->uploadMaterial(m_selection.selected_mesh_index, m_selection.editor_material);
		}

		m_selection.camera.fov_deg = std::clamp(m_selection.camera.fov_deg, 20.0f, 120.0f);
		m_fly_camera->m_fov        = m_selection.camera.fov_deg;
		m_scene.m_camera.m_fov     = m_selection.camera.fov_deg;

		if (m_controls.reset_water_requested) {
			if (m_water_surface != nullptr) {
				m_water_surface->requestReset();
			}
			m_controls.reset_water_requested = false;
		}

		if (m_controls.reset_objects_requested) {
			if (m_water_surface != nullptr) {
				m_water_surface->requestObjectReset();
			}
			m_controls.reset_objects_requested = false;
		}

		if (m_show_all_gui && m_controls.show_overlay) {
			ui::drawAudienceOverlay(ImGui::GetIO().DisplaySize, m_controls.overlay,
			                        m_controls.style);
		}

		const bool camera_moving_this_frame = m_fly_camera->update(m_window->handle(), delta_time);
		const GPUCamera gpu_camera          = m_fly_camera->pack();
		syncSceneCamera(gpu_camera);
		m_backend->updateCamera(gpu_camera);

		const int current_lmb = glfwGetMouseButton(m_window->handle(), GLFW_MOUSE_BUTTON_LEFT);
		if (current_lmb == GLFW_PRESS && m_prev_lmb == GLFW_RELEASE &&
		    !m_fly_camera->isMouseCaptured() && !ImGui::GetIO().WantCaptureMouse) {
			if (ui::selectMesh(m_selection, &m_scene,
			                   pickMeshAtCursor(&m_scene, m_window->handle(), gpu_camera,
			                                    framebuffer_width, framebuffer_height))) {
				selection_panel_result.selection_changed = true;
			}
		}
		m_prev_lmb = current_lmb;

		ImGui::Render();

		const vulkan::RuntimeFrameInput frame_input{
		    m_selection,
		    m_controls.render_post,
		    selection_panel_result.material_edit_active,
		    selection_panel_result.material_changed,
		    selection_panel_result.material_edit_just_finished,
		    selection_panel_result.selection_changed,
		    selection_panel_result.hipr_settings_changed,
		    selection_panel_result.path_tracing_settings_changed,
		    camera_moving_this_frame,
		};
		m_backend->renderFrame(frame_input);
	}

  private:
	void handleHotkeys() {
		const int f1 = glfwGetKey(m_window->handle(), GLFW_KEY_F1);
		if (f1 == GLFW_PRESS && m_prev_f1 == GLFW_RELEASE) {
			m_show_all_gui = !m_show_all_gui;
			if (m_show_all_gui) {
				m_show_control_panel   = true;
				m_show_selection_panel = true;
			}
		}
		m_prev_f1 = f1;

		const int f11 = glfwGetKey(m_window->handle(), GLFW_KEY_F11);
		if (f11 == GLFW_PRESS && m_prev_f11 == GLFW_RELEASE) {
			m_window->toggle_fullscreen();
		}
		m_prev_f11 = f11;

		const int f6 = glfwGetKey(m_window->handle(), GLFW_KEY_F6);
		if (f6 == GLFW_PRESS && m_prev_f6 == GLFW_RELEASE) {
			m_backend->rebuildPipeline(m_selection.debug_view_mode);
		}
		m_prev_f6 = f6;
	}

	void updateAudio(const audio::ReactiveAudioInputFrame& audio_frame, float delta_time) {
		const float audio_level    = m_audio_controller.update(m_controls.audio, audio_frame);
		m_diagnostics.audio        = m_audio_controller.diagnostics();
		m_selection_voice_loudness = std::clamp(audio_level, 0.0f, 1.0f);
		applyOverlayLevel(m_controls, audio_level);

		if (m_water_surface != nullptr) {
			m_diagnostics.water = m_water_surface->prepareFrame(
			    m_controls.water, m_diagnostics.audio.normalized_level, audio_frame.time_seconds,
			    delta_time);
		} else {
			m_diagnostics.water = {};
		}
	}

	void recreateWaterSurface() {
		const simulation::WaterSurfaceCreateInfo create_info = m_backend->waterSurfaceCreateInfo();
		if (create_info.output_extent.width == 0 || create_info.output_extent.height == 0) {
			m_water_surface.reset();
			m_diagnostics.water = {};
			return;
		}

		m_water_surface     = std::make_unique<simulation::WaterSurfaceSimulation>(create_info);
		m_diagnostics.water = {};
	}

	void sanitizeSelectionState(ui::SelectionPanelResult& selection_panel_result) {
		const uint32_t sanitized_rank_count =
		    std::clamp(m_selection.hipr_debug.rank_count, 1u, kMaxHiPRTopK);
		const uint32_t sanitized_frames =
		    sanitizeHiPRFramesPerObject(m_selection.hipr_debug.frames_per_object);
		const uint32_t sanitized_spp = std::clamp(m_selection.path_tracing.spp, 1u, 1024u);
		const uint32_t sanitized_max_bounces =
		    std::clamp(m_selection.path_tracing.max_bounces, 1u, 1024u);

		if (sanitized_rank_count != m_selection.hipr_debug.rank_count) {
			m_selection.hipr_debug.rank_count            = sanitized_rank_count;
			selection_panel_result.hipr_settings_changed = true;
		}
		if (sanitized_frames != m_selection.hipr_debug.frames_per_object) {
			m_selection.hipr_debug.frames_per_object     = sanitized_frames;
			selection_panel_result.hipr_settings_changed = true;
		}
		if (sanitized_spp != m_selection.path_tracing.spp) {
			m_selection.path_tracing.spp                         = sanitized_spp;
			selection_panel_result.path_tracing_settings_changed = true;
		}
		if (sanitized_max_bounces != m_selection.path_tracing.max_bounces) {
			m_selection.path_tracing.max_bounces                 = sanitized_max_bounces;
			selection_panel_result.path_tracing_settings_changed = true;
		}
	}

	void syncSceneCamera(const GPUCamera& camera) {
		m_scene.m_camera.m_position = glm::vec3(camera.position);
		m_scene.m_camera.m_target   = glm::vec3(camera.target);
		m_scene.m_camera.m_up       = glm::vec3(camera.up);
		m_scene.m_camera.m_fov      = camera.fov_near_far.x;
		m_scene.m_camera.m_nearClip = camera.fov_near_far.y;
		m_scene.m_camera.m_farClip  = camera.fov_near_far.z;
	}

	core::Window*                                       m_window = nullptr;
	Scene                                               m_scene{};
	std::unique_ptr<vulkan::Runtime>                    m_backend;
	std::unique_ptr<simulation::WaterSurfaceSimulation> m_water_surface;
	audio::ReactiveAudioController                      m_audio_controller;
	std::unique_ptr<FlyCamera>                          m_fly_camera;
	ui::SelectionContext                                m_selection{};
	ui::AudienceControlPanelState                       m_controls{};
	ui::AudienceDiagnostics                             m_diagnostics{};
	float                                               m_selection_voice_loudness = 0.0f;
	bool                                                m_show_all_gui             = true;
	bool                                                m_show_selection_panel     = true;
	bool                                                m_show_control_panel       = true;
	int                                                 m_prev_f1                  = GLFW_RELEASE;
	int                                                 m_prev_f6                  = GLFW_RELEASE;
	int                                                 m_prev_f11                 = GLFW_RELEASE;
	int                                                 m_prev_lmb                 = GLFW_RELEASE;
};

Renderer::Renderer(core::Window& window, const std::string& scene_argument) :
    m_impl(std::make_unique<Impl>(window, scene_argument)) {
}

Renderer::~Renderer() = default;

void Renderer::tick(float time_seconds, float delta_time,
                    const audio::ReactiveAudioInputFrame& audio_frame) {
	m_impl->tick(time_seconds, delta_time, audio_frame);
}

}        // namespace renderer
