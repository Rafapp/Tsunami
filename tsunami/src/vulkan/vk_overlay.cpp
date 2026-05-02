#include "tsunami/vulkan/vk_overlay.h"

#include <stdexcept>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "tsunami/vulkan/internal/vk_context.h"
#include "tsunami/vulkan/vk_helpers.h"

void create_overlay_render_pass() {
	VkAttachmentDescription color_attachment{};
	color_attachment.format         = swapchain_ctx.image_format;
	color_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
	color_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref{};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments    = &color_attachment_ref;

	VkSubpassDependency dependency{};
	dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass    = 0;
	dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstAccessMask =
	    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo render_pass_info{};
	render_pass_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments    = &color_attachment;
	render_pass_info.subpassCount    = 1;
	render_pass_info.pSubpasses      = &subpass;
	render_pass_info.dependencyCount = 1;
	render_pass_info.pDependencies   = &dependency;

	if (vkCreateRenderPass(vulkan_ctx.device, &render_pass_info, nullptr,
	                       &overlay_ctx.render_pass) != VK_SUCCESS)
		throw std::runtime_error("failed to create overlay render pass");

	overlay_ctx.framebuffers.resize(swapchain_ctx.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain_ctx.image_views.size(); ++i) {
		VkImageView attachments[] = {swapchain_ctx.image_views[i]};

		VkFramebufferCreateInfo framebuffer_info{};
		framebuffer_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_info.renderPass      = overlay_ctx.render_pass;
		framebuffer_info.attachmentCount = 1;
		framebuffer_info.pAttachments    = attachments;
		framebuffer_info.width           = swapchain_ctx.extent.width;
		framebuffer_info.height          = swapchain_ctx.extent.height;
		framebuffer_info.layers          = 1;

		if (vkCreateFramebuffer(vulkan_ctx.device, &framebuffer_info, nullptr,
		                        &overlay_ctx.framebuffers[i]) != VK_SUCCESS)
			throw std::runtime_error("failed to create overlay framebuffer");
	}
}

void destroy_overlay_render_resources() {
	for (VkFramebuffer fb : overlay_ctx.framebuffers) {
		if (fb != VK_NULL_HANDLE)
			vkDestroyFramebuffer(vulkan_ctx.device, fb, nullptr);
	}
	overlay_ctx.framebuffers.clear();

	if (overlay_ctx.render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(vulkan_ctx.device, overlay_ctx.render_pass, nullptr);
		overlay_ctx.render_pass = VK_NULL_HANDLE;
	}
}

void initialize_imgui_context(GLFWwindow* window) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io    = ImGui::GetIO();
	io.IniFilename = nullptr;

	ImGui::StyleColorsDark();

	if (!ImGui_ImplGlfw_InitForVulkan(window, true))
		throw std::runtime_error("failed to initialize ImGui for Vulkan");
}

void initialize_imgui_renderer() {
	ImGui_ImplVulkan_InitInfo init_info{};
	init_info.ApiVersion                   = VK_API_VERSION_1_3;
	init_info.Instance                     = vulkan_ctx.instance.instance;
	init_info.PhysicalDevice               = vulkan_ctx.phys_device.physical_device;
	init_info.Device                       = vulkan_ctx.device;
	init_info.QueueFamily                  = vulkan_ctx.graphics_queue_family;
	init_info.Queue                        = vulkan_ctx.graphics_queue;
	init_info.DescriptorPoolSize           = 16;
	init_info.MinImageCount                = static_cast<uint32_t>(swapchain_ctx.images.size());
	init_info.ImageCount                   = static_cast<uint32_t>(swapchain_ctx.images.size());
	init_info.CheckVkResultFn              = check_vk_result;
	init_info.MinAllocationSize            = 1024 * 1024;
	init_info.PipelineInfoMain.RenderPass  = overlay_ctx.render_pass;
	init_info.PipelineInfoMain.Subpass     = 0;
	init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	if (!ImGui_ImplVulkan_Init(&init_info))
		throw std::runtime_error("failed to initialize ImGui Vulkan backend");
}

void shutdown_imgui_renderer() {
	if (ImGui::GetCurrentContext() == nullptr)
		return;
	ImGui_ImplVulkan_Shutdown();
}

void shutdown_imgui() {
	if (ImGui::GetCurrentContext() == nullptr)
		return;
	shutdown_imgui_renderer();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}
