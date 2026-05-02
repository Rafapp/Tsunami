#pragma once

#include <memory>

#include "tsunami/ui/render_settings.h"
#include "tsunami/ui/selection_panel.h"

struct GPUMaterial;
struct GPUCamera;

class Scene;

namespace ui {
enum class RenderDebugViewMode : int;
}

namespace core {
class Window;
}

namespace simulation {
struct WaterSurfaceCreateInfo;
}

namespace vulkan {

struct RuntimeFrameInput {
	const ui::SelectionContext&           selection;
	const ui::AudienceRenderPostSettings& render_post;
	bool                                  material_edit_active          = false;
	bool                                  material_changed              = false;
	bool                                  material_edit_just_finished   = false;
	bool                                  selection_changed             = false;
	bool                                  hipr_settings_changed         = false;
	bool                                  path_tracing_settings_changed = false;
	bool                                  camera_moving                 = false;
};

struct RuntimeFrameOutput {
	ui::AudienceRenderDiagnostics render{};
	bool                          surface_resized = false;
};

class Runtime {
  public:
	Runtime(core::Window& window, const Scene& scene);
	~Runtime();

	Runtime(const Runtime&)            = delete;
	Runtime& operator=(const Runtime&) = delete;
	Runtime(Runtime&&)                 = delete;
	Runtime& operator=(Runtime&&)      = delete;

	RuntimeFrameOutput                 beginFrame();
	void                               renderFrame(const RuntimeFrameInput& input);
	simulation::WaterSurfaceCreateInfo waterSurfaceCreateInfo() const;
	void uploadMaterial(int material_index, const GPUMaterial& material);
	bool rebuildPipeline(ui::RenderDebugViewMode mode);
	void updateCamera(const GPUCamera& camera);
	void waitIdle();

  private:
	class Impl;
	std::unique_ptr<Impl> m_impl;
};

}        // namespace vulkan
