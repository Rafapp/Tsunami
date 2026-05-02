#include "tsunami/vulkan/vk_pipeline.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "slang.h"

// Mode → filename/label tables and shader compilation live here.
// Each mode gets its own VkPipelineLayout so push constant structs
// can diverge independently.

// ============================================================
// Mode tables
// ============================================================
const std::array<ui::RenderDebugViewMode, 4> kRenderModes = {
    ui::RenderDebugViewMode::HiPR,
    ui::RenderDebugViewMode::Naive,
    ui::RenderDebugViewMode::HiPRVis,
    ui::RenderDebugViewMode::ObjectIds,
};

static const char* mode_shader_filename(ui::RenderDebugViewMode mode) {
	switch (mode) {
		case ui::RenderDebugViewMode::HiPR:
			return "hipr.slang";
		case ui::RenderDebugViewMode::Naive:
			return "naivept.slang";
		case ui::RenderDebugViewMode::HiPRVis:
			return "hiprvis.slang";
		case ui::RenderDebugViewMode::ObjectIds:
			return "objectid.slang";
	}
	return "hipr.slang";
}

static const char* mode_label(ui::RenderDebugViewMode mode) {
	switch (mode) {
		case ui::RenderDebugViewMode::HiPR:
			return "HiPR";
		case ui::RenderDebugViewMode::Naive:
			return "NaivePT";
		case ui::RenderDebugViewMode::HiPRVis:
			return "HiPRVis";
		case ui::RenderDebugViewMode::ObjectIds:
			return "ObjectID";
	}
	return "HiPR";
}

static uint32_t mode_push_constant_size(ui::RenderDebugViewMode mode) {
	switch (mode) {
		case ui::RenderDebugViewMode::HiPR:
			return sizeof(HiPRPushConstants);
		case ui::RenderDebugViewMode::Naive:
			return sizeof(NaivePTPushConstants);
		case ui::RenderDebugViewMode::HiPRVis:
			return sizeof(HiPRVisPushConstants);
		case ui::RenderDebugViewMode::ObjectIds:
			return sizeof(ObjectIdPushConstants);
	}
	return sizeof(PathTracerPushConstants);
}

size_t render_mode_index(ui::RenderDebugViewMode mode) {
	switch (mode) {
		case ui::RenderDebugViewMode::HiPR:
			return 0;
		case ui::RenderDebugViewMode::Naive:
			return 1;
		case ui::RenderDebugViewMode::HiPRVis:
			return 2;
		case ui::RenderDebugViewMode::ObjectIds:
			return 3;
	}
	return 0;
}

// ============================================================
// Descriptor layout helper
// ============================================================
VkDescriptorSetLayoutBinding make_binding(uint32_t binding, VkDescriptorType type, uint32_t count) {
	VkDescriptorSetLayoutBinding b{};
	b.binding         = binding;
	b.descriptorType  = type;
	b.descriptorCount = count;
	b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	return b;
}

// ============================================================
// Pipeline lifecycle
// ============================================================
void init_pipeline_modes(VkDevice device, VkDescriptorSetLayout desc_layout) {
	for (size_t i = 0; i < kRenderModes.size(); ++i) {
		const ui::RenderDebugViewMode m  = kRenderModes[i];
		PipelineMode&                 pm = compute_ctx.modes[i];
		pm.shader_file                   = mode_shader_filename(m);
		pm.label                         = mode_label(m);
		pm.push_constant_size            = mode_push_constant_size(m);

		VkPushConstantRange pr{};
		pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pr.size       = pm.push_constant_size;

		VkPipelineLayoutCreateInfo pli{};
		pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount         = 1;
		pli.pSetLayouts            = &desc_layout;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges    = &pr;

		if (vkCreatePipelineLayout(device, &pli, nullptr, &pm.layout) != VK_SUCCESS)
			throw std::runtime_error(std::string("failed to create pipeline layout for mode: ") +
			                         pm.label);
	}
}

void destroy_pipeline_modes(VkDevice device) {
	for (auto& pm : compute_ctx.modes) {
		if (pm.pipeline != VK_NULL_HANDLE) {
			vkDestroyPipeline(device, pm.pipeline, nullptr);
			pm.pipeline = VK_NULL_HANDLE;
		}
		pm.compiled = false;
		if (pm.layout != VK_NULL_HANDLE) {
			vkDestroyPipelineLayout(device, pm.layout, nullptr);
			pm.layout = VK_NULL_HANDLE;
		}
	}
}

// ============================================================
// Shader compilation
// ============================================================
static std::vector<uint32_t>
    compile_slang_shader(const std::string& path, const std::string& entry_point,
                         const std::vector<std::string>& search_paths = {}) {
	SlangSession*        session = spCreateSession(nullptr);
	SlangCompileRequest* req     = spCreateCompileRequest(session);
	for (const auto& sp : search_paths)
		spAddSearchPath(req, sp.c_str());
	int ti = spAddCodeGenTarget(req, SLANG_SPIRV);
	spSetTargetProfile(req, ti, spFindProfile(session, "spirv_1_4"));
	int ui = spAddTranslationUnit(req, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
	spAddTranslationUnitSourceFile(req, ui, path.c_str());
	spAddEntryPoint(req, ui, entry_point.c_str(), SLANG_STAGE_COMPUTE);
	SlangResult res  = spCompile(req);
	const char* diag = spGetDiagnosticOutput(req);
	if (diag && diag[0] != '\0')
		std::cerr << "[SLANG] " << path << ":\n" << diag << "\n";
	if (res != SLANG_OK) {
		spDestroyCompileRequest(req);
		spDestroySession(session);
		throw std::runtime_error("slang compilation failed: " + path);
	}
	size_t                sz   = 0;
	const void*           data = spGetEntryPointCode(req, 0, &sz);
	std::vector<uint32_t> spirv(sz / sizeof(uint32_t));
	memcpy(spirv.data(), data, sz);
	spDestroyCompileRequest(req);
	spDestroySession(session);
	return spirv;
}

bool create_pipeline_for_mode(ui::RenderDebugViewMode mode, VkPipeline& out_pipeline,
                              size_t* out_shader_size_bytes) {
	const size_t           idx    = render_mode_index(mode);
	const char*            label  = compute_ctx.modes[idx].label;
	const VkPipelineLayout layout = compute_ctx.modes[idx].layout;

	std::vector<uint32_t> spirv;
	const std::string     shader_path =
	    std::string(SHADERS_DIR) + "/" + compute_ctx.modes[idx].shader_file;
	try {
		spirv = compile_slang_shader(shader_path, "main", {VENDORS_DIR});
	} catch (const std::exception& e) {
		std::cerr << "[PIPELINE] " << label << " compile failed: " << e.what() << "\n";
		return false;
	}

	if (out_shader_size_bytes)
		*out_shader_size_bytes = spirv.size() * sizeof(uint32_t);

	VkShaderModuleCreateInfo mci{};
	mci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	mci.codeSize = spirv.size() * sizeof(uint32_t);
	mci.pCode    = spirv.data();

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(vulkan_ctx.device, &mci, nullptr, &shader_module) != VK_SUCCESS) {
		std::cerr << "[PIPELINE] " << label << " vkCreateShaderModule failed\n";
		return false;
	}

	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = shader_module;
	stage.pName  = "main";

	VkComputePipelineCreateInfo pci{};
	pci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pci.stage  = stage;
	pci.layout = layout;

	VkPipeline new_pipeline = VK_NULL_HANDLE;
	if (vkCreateComputePipelines(vulkan_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr,
	                             &new_pipeline) != VK_SUCCESS) {
		vkDestroyShaderModule(vulkan_ctx.device, shader_module, nullptr);
		std::cerr << "[PIPELINE] " << label << " vkCreateComputePipelines failed\n";
		return false;
	}

	vkDestroyShaderModule(vulkan_ctx.device, shader_module, nullptr);
	out_pipeline = new_pipeline;
	return true;
}

bool ensure_pipeline_for_mode(ui::RenderDebugViewMode mode) {
	PipelineMode& pm = compute_ctx.modes[render_mode_index(mode)];
	if (pm.compiled && pm.pipeline != VK_NULL_HANDLE)
		return true;

	size_t shader_size = 0;
	if (!create_pipeline_for_mode(mode, pm.pipeline, &shader_size))
		return false;

	pm.compiled = true;
	std::cout << "[PIPELINE] Built " << pm.label << " ("
	          << static_cast<unsigned long long>(shader_size) << " bytes SPIR-V)\n";
	return true;
}

bool build_all_mode_pipelines() {
	for (ui::RenderDebugViewMode m : kRenderModes) {
		if (!ensure_pipeline_for_mode(m))
			return false;
	}
	return true;
}

bool rebuild_pipeline(ui::RenderDebugViewMode mode) {
	PipelineMode& pm = compute_ctx.modes[render_mode_index(mode)];

	VkPipeline new_pipeline = VK_NULL_HANDLE;
	size_t     shader_size  = 0;
	if (!create_pipeline_for_mode(mode, new_pipeline, &shader_size))
		return false;

	vkDeviceWaitIdle(vulkan_ctx.device);
	if (pm.pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(vulkan_ctx.device, pm.pipeline, nullptr);
	}
	pm.pipeline = new_pipeline;
	pm.compiled = true;

	std::cout << "[SHADER RELOAD] Rebuilt " << pm.label << " ("
	          << static_cast<unsigned long long>(shader_size) << " bytes SPIR-V)\n";
	return true;
}
