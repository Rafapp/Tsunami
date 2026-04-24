#include "tsunami/ui/imgui_runtime.h"

#include <stdexcept>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace ui {

void createOverlayRenderPassAndFramebuffers(VkDevice device, VkFormat image_format,
                                            VkExtent2D                      extent,
                                            const std::vector<VkImageView>& image_views,
                                            VkRenderPass&                   render_pass,
                                            std::vector<VkFramebuffer>&     framebuffers) {
	VkAttachmentDescription color_attachment{};
	color_attachment.format         = image_format;
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

	if (vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create overlay render pass");
	}

	framebuffers.resize(image_views.size(), VK_NULL_HANDLE);
	for (size_t index = 0; index < image_views.size(); ++index) {
		VkImageView attachments[] = {image_views[index]};

		VkFramebufferCreateInfo framebuffer_info{};
		framebuffer_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_info.renderPass      = render_pass;
		framebuffer_info.attachmentCount = 1;
		framebuffer_info.pAttachments    = attachments;
		framebuffer_info.width           = extent.width;
		framebuffer_info.height          = extent.height;
		framebuffer_info.layers          = 1;

		if (vkCreateFramebuffer(device, &framebuffer_info, nullptr, &framebuffers[index]) !=
		    VK_SUCCESS) {
			throw std::runtime_error("failed to create overlay framebuffer");
		}
	}
}

void recreateOverlayFramebuffers(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
                                 const std::vector<VkImageView>& image_views,
                                 std::vector<VkFramebuffer>&     framebuffers) {
	for (VkFramebuffer framebuffer : framebuffers) {
		if (framebuffer != VK_NULL_HANDLE) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
	}
	framebuffers.clear();
	framebuffers.resize(image_views.size(), VK_NULL_HANDLE);

	for (size_t index = 0; index < image_views.size(); ++index) {
		VkImageView attachments[] = {image_views[index]};

		VkFramebufferCreateInfo framebuffer_info{};
		framebuffer_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_info.renderPass      = render_pass;
		framebuffer_info.attachmentCount = 1;
		framebuffer_info.pAttachments    = attachments;
		framebuffer_info.width           = extent.width;
		framebuffer_info.height          = extent.height;
		framebuffer_info.layers          = 1;

		if (vkCreateFramebuffer(device, &framebuffer_info, nullptr, &framebuffers[index]) !=
		    VK_SUCCESS) {
			throw std::runtime_error("failed to recreate overlay framebuffer");
		}
	}
}

void destroyOverlayRenderResources(VkDevice device, VkRenderPass& render_pass,
                                   std::vector<VkFramebuffer>& framebuffers) {
	for (VkFramebuffer framebuffer : framebuffers) {
		if (framebuffer != VK_NULL_HANDLE) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
	}
	framebuffers.clear();

	if (render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}
}

void initializeImGuiContext(GLFWwindow* window) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io    = ImGui::GetIO();
	io.IniFilename = nullptr;

	ImGui::StyleColorsDark();

	if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
		throw std::runtime_error("failed to initialize ImGui for Vulkan");
	}
}

void initializeImGuiRenderer(const ImGuiRendererInitInfo& init_info) {
	ImGui_ImplVulkan_InitInfo data{};
	data.ApiVersion                   = VK_API_VERSION_1_3;
	data.Instance                     = init_info.instance;
	data.PhysicalDevice               = init_info.physical_device;
	data.Device                       = init_info.device;
	data.QueueFamily                  = init_info.graphics_queue_family;
	data.Queue                        = init_info.graphics_queue;
	data.DescriptorPoolSize           = 16;
	data.MinImageCount                = init_info.image_count;
	data.ImageCount                   = init_info.image_count;
	data.CheckVkResultFn              = init_info.check_vk_result;
	data.MinAllocationSize            = 1024 * 1024;
	data.PipelineInfoMain.RenderPass  = init_info.render_pass;
	data.PipelineInfoMain.Subpass     = 0;
	data.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	if (!ImGui_ImplVulkan_Init(&data)) {
		throw std::runtime_error("failed to initialize ImGui Vulkan backend");
	}
}

void shutdownImGuiRenderer() {
	if (ImGui::GetCurrentContext() == nullptr) {
		return;
	}

	ImGui_ImplVulkan_Shutdown();
}

void shutdownImGui() {
	if (ImGui::GetCurrentContext() == nullptr) {
		return;
	}

	shutdownImGuiRenderer();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

}        // namespace ui
