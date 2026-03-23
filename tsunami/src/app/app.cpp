#include <iostream>

#include "tsunami/app.h"

struct VulkanContext {
    // core
    vkb::Instance       instance;
    vkb::PhysicalDevice phys_device;
	vkb::Device         log_device;
    VkSurfaceKHR        surface;

    // queues
    VkQueue  graphics_queue;
    uint32_t graphics_queue_family;
};

struct SwapchainContext {
    vkb::Swapchain           swapchain;
    std::vector<VkImage>     images;
    std::vector<VkImageView> image_views;
    VkFormat                 image_format;
    VkExtent2D               extent;
};

struct RenderContext {
    VkRenderPass               render_pass;
    std::vector<VkFramebuffer> framebuffers;
};

App::App() {
	// === I. Load vulkan function pointers ===
	if (volkInitialize() != VK_SUCCESS) {
		throw std::runtime_error("failed to initialize volk");
	}

	// === II. Create window ===
	m_window = std::make_unique<core::Window>(
	    core::WindowConfig{.width = 1280, .height = 720, .title = "tsunami 🌊"});

	// === III. Create Vulkan context ===
	VulkanContext vulkan_ctx;

	// 1. Create Vulkan instance
	vkb::InstanceBuilder builder;
	auto inst_ret = builder.set_app_name ("Example Vulkan Application")
					.request_validation_layers ()
					.use_default_debug_messenger ()
					.build ();
    if (!inst_ret) throw std::runtime_error("failed to create Vulkan instance");
	vulkan_ctx.instance = inst_ret.value ();
	volkLoadInstance(vulkan_ctx.instance.instance);

	// 3. Create a surface
	vulkan_ctx.surface = VK_NULL_HANDLE;
	if (glfwCreateWindowSurface(vulkan_ctx.instance, m_window->handle(), nullptr, &vulkan_ctx.surface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface");
	}

	// 5. Select physical device (GPU)
	auto phys_dev_ret = vkb::PhysicalDeviceSelector(vulkan_ctx.instance)
		.set_surface(vulkan_ctx.surface)
		.set_minimum_version(1, 0)
		.select();
	if (!phys_dev_ret) throw std::runtime_error("failed to select physical device");
	vulkan_ctx.phys_device = phys_dev_ret.value();

	// 6. Create logical device and load with volk
	vkb::DeviceBuilder device_builder{ vulkan_ctx.phys_device };
	auto dev_ret = device_builder.build ();
	if (!dev_ret) throw std::runtime_error("failed to create logical device");
	vulkan_ctx.log_device = dev_ret.value ();
	volkLoadDevice(vulkan_ctx.log_device.device);

	// 7. Create graphics queue
	auto graphics_queue_ret = vulkan_ctx.log_device.get_queue (vkb::QueueType::graphics);
    if (!graphics_queue_ret) throw std::runtime_error("failed to get graphics queue");
    VkQueue graphics_queue = graphics_queue_ret.value ();

	// === IV. Create swapchain ===
	SwapchainContext swapchain_ctx;

	// 1. Create swapchain
	vkb::SwapchainBuilder swapchain_builder{ vulkan_ctx.log_device };
    auto swap_ret = swapchain_builder
		.set_desired_format({ VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(m_window->width(), m_window->height())
		.build();
	if (!swap_ret) throw std::runtime_error("failed to create swapchain");
	swapchain_ctx.swapchain = swap_ret.value();
	swapchain_ctx.image_format = swapchain_ctx.swapchain.image_format;

	// 2. Create swapchain images
	auto images_ret = swapchain_ctx.swapchain.get_images();
	if (!images_ret) throw std::runtime_error("failed to get swapchain images");
	swapchain_ctx.images = images_ret.value();

	// 3. Create swapchain image views
	auto image_views_ret = swapchain_ctx.swapchain.get_image_views();
	if (!image_views_ret) throw std::runtime_error("failed to get swapchain image views");
	swapchain_ctx.image_views = image_views_ret.value();

	// === V. Create render context === 
	RenderContext render_ctx;

	// 1. Create render pass
	VkAttachmentDescription color_attachment = {};
    color_attachment.format = swapchain_ctx.image_format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &color_attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

	if (vkCreateRenderPass(vulkan_ctx.log_device.device, &render_pass_info, nullptr, &render_ctx.render_pass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass");
	}
}

void App::run() {
	MainLoop();
}

void App::MainLoop() {
	while (!m_window->shouldClose()) {
		m_window->pollEvents();
	}
}