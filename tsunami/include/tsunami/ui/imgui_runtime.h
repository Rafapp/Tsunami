#pragma once

#include <cstdint>
#include <vector>

#ifndef VK_NO_PROTOTYPES
#	define VK_NO_PROTOTYPES
#endif

#include "volk.h"

struct GLFWwindow;

namespace ui {

struct ImGuiRendererInitInfo {
	VkInstance       instance              = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device       = VK_NULL_HANDLE;
	VkDevice         device                = VK_NULL_HANDLE;
	uint32_t         graphics_queue_family = 0;
	VkQueue          graphics_queue        = VK_NULL_HANDLE;
	uint32_t         image_count           = 0;
	VkRenderPass     render_pass           = VK_NULL_HANDLE;
	void (*check_vk_result)(VkResult)      = nullptr;
};

void createOverlayRenderPassAndFramebuffers(VkDevice device, VkFormat image_format,
                                            VkExtent2D                      extent,
                                            const std::vector<VkImageView>& image_views,
                                            VkRenderPass&                   render_pass,
                                            std::vector<VkFramebuffer>&     framebuffers);

void recreateOverlayFramebuffers(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
                                 const std::vector<VkImageView>& image_views,
                                 std::vector<VkFramebuffer>&     framebuffers);

void destroyOverlayRenderResources(VkDevice device, VkRenderPass& render_pass,
                                   std::vector<VkFramebuffer>& framebuffers);

void initializeImGuiContext(GLFWwindow* window);
void initializeImGuiRenderer(const ImGuiRendererInitInfo& init_info);
void shutdownImGuiRenderer();
void shutdownImGui();

}        // namespace ui
