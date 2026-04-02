#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define VOLK_IMPLEMENTATION
#include "volk.h"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "VkBootstrap.h"
#include "vk_mem_alloc.h"

#include "tsunami/app/app.h"
#include "tsunami/audio/microphone_input.h"
#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_surface_simulation.h"
#include "tsunami/ui/audience_control_panel.h"
#include "tsunami/ui/audience_overlay.h"

namespace {

struct VulkanContext {
	vkb::Instance       instance{};
	vkb::PhysicalDevice phys_device{};
	vkb::Device         log_device{};
	VkDevice            device                = VK_NULL_HANDLE;
	VkSurfaceKHR        surface               = VK_NULL_HANDLE;
	VkQueue             graphics_queue        = VK_NULL_HANDLE;
	uint32_t            graphics_queue_family = 0;
} vulkan_ctx{};

struct SwapchainContext {
	vkb::Swapchain           swapchain{};
	std::vector<VkImage>     images;
	std::vector<VkImageView> image_views;
	std::vector<bool>        image_initialized;
	VkFormat                 image_format = VK_FORMAT_UNDEFINED;
	VkExtent2D               extent{};
} swapchain_ctx{};

struct RenderResourcesContext {
	VmaAllocator allocator = nullptr;
} render_resources_ctx{};

struct CommandContext {
	VkCommandPool   command_pool   = VK_NULL_HANDLE;
	VkCommandBuffer command_buffer = VK_NULL_HANDLE;
} command_ctx{};

struct SyncContext {
	VkSemaphore image_available = VK_NULL_HANDLE;
	VkSemaphore render_finished = VK_NULL_HANDLE;
	VkFence     in_flight       = VK_NULL_HANDLE;
} sync_ctx{};

struct OverlayContext {
	VkRenderPass                  render_pass = VK_NULL_HANDLE;
	std::vector<VkFramebuffer>    framebuffers;
	ui::AudienceControlPanelState controls{};
	ui::AudienceDiagnostics       diagnostics{};
	bool                          show_control_panel = true;
} overlay_ctx{};

void check_vk_result(VkResult result) {
	if (result == VK_SUCCESS) {
		return;
	}

	std::cerr << "[Vulkan] VkResult = " << result << "\n";
	if (result < 0) {
		throw std::runtime_error("Vulkan call failed");
	}
}

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
	                       &overlay_ctx.render_pass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create overlay render pass");
	}

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
		                        &overlay_ctx.framebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create overlay framebuffer");
		}
	}
}

void destroy_overlay_render_resources() {
	for (VkFramebuffer framebuffer : overlay_ctx.framebuffers) {
		if (framebuffer != VK_NULL_HANDLE) {
			vkDestroyFramebuffer(vulkan_ctx.device, framebuffer, nullptr);
		}
	}
	overlay_ctx.framebuffers.clear();

	if (overlay_ctx.render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(vulkan_ctx.device, overlay_ctx.render_pass, nullptr);
		overlay_ctx.render_pass = VK_NULL_HANDLE;
	}
}

void initialize_imgui(GLFWwindow* window) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io    = ImGui::GetIO();
	io.IniFilename = nullptr;

	ImGui::StyleColorsDark();

	if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
		throw std::runtime_error("failed to initialize ImGui for Vulkan");
	}

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

	if (!ImGui_ImplVulkan_Init(&init_info)) {
		throw std::runtime_error("failed to initialize ImGui Vulkan backend");
	}
}

void shutdown_imgui() {
	if (ImGui::GetCurrentContext() == nullptr) {
		return;
	}

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

audio::ReactiveAudioInputFrame buildAudioInputFrame(const audio::MicrophoneInput* microphone,
                                                    float                         time_seconds) {
	audio::ReactiveAudioInputFrame input_frame{};
	input_frame.source_available = microphone != nullptr && microphone->isAvailable();
	input_frame.raw_level        = input_frame.source_available ? microphone->latestLevel() : 0.0f;
	input_frame.time_seconds     = time_seconds;
	input_frame.source_name =
	    microphone != nullptr ? microphone->deviceName() : std::string("Unavailable");
	input_frame.source_status = microphone != nullptr ?
	                                microphone->statusMessage() :
	                                std::string("Microphone capture is unavailable.");
	return input_frame;
}

void applyOverlayLevel(float value) {
	overlay_ctx.controls.overlay.volume_level   = std::clamp(value, 0.0f, 1.0f);
	overlay_ctx.controls.overlay.selected_index = ui::quantizeSelection(
	    overlay_ctx.controls.overlay.volume_level, overlay_ctx.controls.overlay.selection_count);
}

}        // namespace

App::App() {
	if (volkInitialize() != VK_SUCCESS) {
		throw std::runtime_error("failed to initialize volk");
	}
	std::cout << "[INFO] Initialized volk\n";

	m_window = std::make_unique<core::Window>(
	    core::WindowConfig{.width = 1280, .height = 720, .title = "tsunami 🌊"});
	std::cout << "[INFO] Created window\n";

	vkb::InstanceBuilder builder;
	auto                 inst_ret = builder.set_app_name("tsunami")
	                                    .request_validation_layers()
	                                    .use_default_debug_messenger()
	                                    .require_api_version(1, 3, 0)
	                                    .build();
	if (!inst_ret) {
		throw std::runtime_error("failed to create Vulkan instance");
	}
	vulkan_ctx.instance = inst_ret.value();
	volkLoadInstance(vulkan_ctx.instance.instance);
	std::cout << "[INFO] Created Vulkan instance\n";

	if (glfwCreateWindowSurface(vulkan_ctx.instance, m_window->handle(), nullptr,
	                            &vulkan_ctx.surface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface");
	}
	std::cout << "[INFO] Created window surface\n";

	auto phys_dev_ret = vkb::PhysicalDeviceSelector(vulkan_ctx.instance)
	                        .set_surface(vulkan_ctx.surface)
	                        .set_minimum_version(1, 3)
	                        .select();
	if (!phys_dev_ret) {
		throw std::runtime_error("failed to select physical device");
	}
	vulkan_ctx.phys_device = phys_dev_ret.value();
	std::cout << "[INFO] Selected physical device\n";

	vkb::DeviceBuilder device_builder{vulkan_ctx.phys_device};
	auto               dev_ret = device_builder.build();
	if (!dev_ret) {
		throw std::runtime_error("failed to create logical device");
	}
	vulkan_ctx.log_device = dev_ret.value();
	vulkan_ctx.device     = vulkan_ctx.log_device.device;
	volkLoadDevice(vulkan_ctx.device);
	std::cout << "[INFO] Created logical device\n";

	auto graphics_queue_ret = vulkan_ctx.log_device.get_queue(vkb::QueueType::graphics);
	if (!graphics_queue_ret) {
		throw std::runtime_error("failed to get graphics queue");
	}
	vulkan_ctx.graphics_queue = graphics_queue_ret.value();
	std::cout << "[INFO] Created graphics queue\n";

	auto family_ret = vulkan_ctx.log_device.get_queue_index(vkb::QueueType::graphics);
	if (!family_ret) {
		throw std::runtime_error("failed to get graphics queue family");
	}
	vulkan_ctx.graphics_queue_family = family_ret.value();

	VmaAllocatorCreateInfo vma_info{};
	vma_info.instance         = vulkan_ctx.instance.instance;
	vma_info.physicalDevice   = vulkan_ctx.phys_device.physical_device;
	vma_info.device           = vulkan_ctx.device;
	vma_info.vulkanApiVersion = VK_API_VERSION_1_3;

	VmaVulkanFunctions vma_vulkan_funcs{};
	vma_vulkan_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	vma_vulkan_funcs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
	vma_info.pVulkanFunctions              = &vma_vulkan_funcs;

	if (vmaCreateAllocator(&vma_info, &render_resources_ctx.allocator) != VK_SUCCESS) {
		throw std::runtime_error("failed to create VMA allocator");
	}
	std::cout << "[INFO] Created VMA allocator\n";

	vkb::SwapchainBuilder swapchain_builder{vulkan_ctx.log_device};
	auto                  swap_ret =
	    swapchain_builder
	        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
	        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
	        .set_desired_extent(m_window->width(), m_window->height())
	        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
	        .build();
	if (!swap_ret) {
		throw std::runtime_error("failed to create swapchain");
	}
	swapchain_ctx.swapchain    = swap_ret.value();
	swapchain_ctx.image_format = swapchain_ctx.swapchain.image_format;
	swapchain_ctx.extent       = swapchain_ctx.swapchain.extent;
	std::cout << "[INFO] Created swapchain (format: " << swapchain_ctx.image_format << ")\n";

	auto images_ret = swapchain_ctx.swapchain.get_images();
	if (!images_ret) {
		throw std::runtime_error("failed to get swapchain images");
	}
	swapchain_ctx.images            = images_ret.value();
	swapchain_ctx.image_initialized = std::vector<bool>(swapchain_ctx.images.size(), false);
	std::cout << "[INFO] Acquired " << swapchain_ctx.images.size() << " swapchain images\n";

	auto image_views_ret = swapchain_ctx.swapchain.get_image_views();
	if (!image_views_ret) {
		throw std::runtime_error("failed to get swapchain image views");
	}
	swapchain_ctx.image_views = image_views_ret.value();
	std::cout << "[INFO] Created swapchain image views\n";

	m_water_surface =
	    std::make_unique<simulation::WaterSurfaceSimulation>(simulation::WaterSurfaceCreateInfo{
	        .device        = vulkan_ctx.device,
	        .allocator     = render_resources_ctx.allocator,
	        .output_extent = {(uint32_t) m_window->width(), (uint32_t) m_window->height()},
	    });
	std::cout << "[INFO] Created water surface simulation resources\n";

	VkCommandPoolCreateInfo cmd_pool_info{};
	cmd_pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmd_pool_info.queueFamilyIndex = vulkan_ctx.graphics_queue_family;
	cmd_pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (vkCreateCommandPool(vulkan_ctx.device, &cmd_pool_info, nullptr,
	                        &command_ctx.command_pool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool");
	}
	std::cout << "[INFO] Created command pool\n";

	VkCommandBufferAllocateInfo cmd_alloc_info{};
	cmd_alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmd_alloc_info.commandPool        = command_ctx.command_pool;
	cmd_alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmd_alloc_info.commandBufferCount = 1;

	if (vkAllocateCommandBuffers(vulkan_ctx.device, &cmd_alloc_info, &command_ctx.command_buffer) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffer");
	}
	std::cout << "[INFO] Allocated command buffer\n";

	VkSemaphoreCreateInfo semaphore_info{};
	semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_info{};
	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr, &sync_ctx.image_available) !=
	        VK_SUCCESS ||
	    vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr, &sync_ctx.render_finished) !=
	        VK_SUCCESS ||
	    vkCreateFence(vulkan_ctx.device, &fence_info, nullptr, &sync_ctx.in_flight) != VK_SUCCESS) {
		throw std::runtime_error("failed to create sync objects");
	}
	std::cout << "[INFO] Created sync objects (image_available, render_finished, in_flight)\n";

	create_overlay_render_pass();
	initialize_imgui(m_window->handle());
	std::cout << "[INFO] Initialized ImGui overlay\n";

	m_audio_controller = std::make_unique<audio::ReactiveAudioController>();
	m_microphone       = std::make_unique<audio::MicrophoneInput>();
	if (m_microphone->isAvailable()) {
		std::cout << "[INFO] Live microphone capture ready: " << m_microphone->deviceName() << "\n";
	} else {
		std::cout << "[WARN] " << m_microphone->statusMessage()
		          << " Falling back to the demo audience signal.\n";
	}
}

App::~App() {
	if (vulkan_ctx.device == VK_NULL_HANDLE) {
		m_water_surface.reset();
		m_audio_controller.reset();
		m_microphone.reset();
		return;
	}

	vkDeviceWaitIdle(vulkan_ctx.device);

	m_water_surface.reset();
	m_audio_controller.reset();
	m_microphone.reset();

	shutdown_imgui();
	destroy_overlay_render_resources();

	if (sync_ctx.in_flight != VK_NULL_HANDLE) {
		vkDestroyFence(vulkan_ctx.device, sync_ctx.in_flight, nullptr);
	}

	if (sync_ctx.render_finished != VK_NULL_HANDLE) {
		vkDestroySemaphore(vulkan_ctx.device, sync_ctx.render_finished, nullptr);
	}

	if (sync_ctx.image_available != VK_NULL_HANDLE) {
		vkDestroySemaphore(vulkan_ctx.device, sync_ctx.image_available, nullptr);
	}

	if (command_ctx.command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(vulkan_ctx.device, command_ctx.command_pool, nullptr);
	}

	if (render_resources_ctx.allocator != VK_NULL_HANDLE) {
		vmaDestroyAllocator(render_resources_ctx.allocator);
	}

	if (!swapchain_ctx.image_views.empty()) {
		swapchain_ctx.swapchain.destroy_image_views(swapchain_ctx.image_views);
	}

	if (swapchain_ctx.swapchain.swapchain != VK_NULL_HANDLE) {
		vkb::destroy_swapchain(swapchain_ctx.swapchain);
	}

	if (vulkan_ctx.surface != VK_NULL_HANDLE) {
		vkb::destroy_surface(vulkan_ctx.instance, vulkan_ctx.surface);
	}

	if (vulkan_ctx.device != VK_NULL_HANDLE) {
		vkb::destroy_device(vulkan_ctx.log_device);
	}

	if (vulkan_ctx.instance.instance != VK_NULL_HANDLE) {
		vkb::destroy_instance(vulkan_ctx.instance);
	}
}

void App::run() {
	MainLoop();
}

void App::MainLoop() {
	float last_frame_time = static_cast<float>(glfwGetTime());

	while (!m_window->shouldClose()) {
		m_window->pollEvents();

		const float time_seconds = static_cast<float>(glfwGetTime());
		const float delta_time   = std::max(time_seconds - last_frame_time, 1.0f / 240.0f);
		last_frame_time          = time_seconds;

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		const audio::ReactiveAudioInputFrame audio_input =
		    buildAudioInputFrame(m_microphone.get(), time_seconds);

		float audio_level = 0.0f;
		if (m_audio_controller != nullptr) {
			audio_level = m_audio_controller->update(overlay_ctx.controls.audio, audio_input);
			overlay_ctx.diagnostics.audio = m_audio_controller->diagnostics();
		}
		const float water_audio_level = overlay_ctx.diagnostics.audio.normalized_level;
		applyOverlayLevel(audio_level);

		if (m_water_surface != nullptr) {
			overlay_ctx.diagnostics.water = m_water_surface->prepareFrame(
			    overlay_ctx.controls.water, water_audio_level, time_seconds, delta_time);
		}

		if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) {
			overlay_ctx.show_control_panel = !overlay_ctx.show_control_panel;
		}

		bool controls_changed = false;
		if (overlay_ctx.show_control_panel) {
			controls_changed = ui::drawAudienceControlPanel(
			    &overlay_ctx.show_control_panel, overlay_ctx.controls, overlay_ctx.diagnostics);
		}

		if (controls_changed) {
			if (m_audio_controller != nullptr) {
				audio_level = m_audio_controller->update(overlay_ctx.controls.audio, audio_input);
				overlay_ctx.diagnostics.audio = m_audio_controller->diagnostics();
			}
			const float updated_water_audio_level = overlay_ctx.diagnostics.audio.normalized_level;
			applyOverlayLevel(audio_level);

			if (m_water_surface != nullptr) {
				overlay_ctx.diagnostics.water = m_water_surface->prepareFrame(
				    overlay_ctx.controls.water, updated_water_audio_level, time_seconds,
				    delta_time);
			}
		}

		if (overlay_ctx.controls.reset_water_requested) {
			if (m_water_surface != nullptr) {
				m_water_surface->requestReset();
			}
			overlay_ctx.controls.reset_water_requested = false;
		}

		if (overlay_ctx.controls.show_overlay) {
			ui::drawAudienceOverlay(ImGui::GetIO().DisplaySize, overlay_ctx.controls.overlay,
			                        overlay_ctx.controls.style);
		}
		ImGui::Render();

		check_vk_result(
		    vkWaitForFences(vulkan_ctx.device, 1, &sync_ctx.in_flight, VK_TRUE, UINT64_MAX));
		check_vk_result(vkResetFences(vulkan_ctx.device, 1, &sync_ctx.in_flight));

		uint32_t image_index = 0;
		check_vk_result(vkAcquireNextImageKHR(vulkan_ctx.device, swapchain_ctx.swapchain.swapchain,
		                                      UINT64_MAX, sync_ctx.image_available, VK_NULL_HANDLE,
		                                      &image_index));

		check_vk_result(vkResetCommandBuffer(command_ctx.command_buffer, 0));

		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		check_vk_result(vkBeginCommandBuffer(command_ctx.command_buffer, &begin_info));

		if (m_water_surface != nullptr) {
			m_water_surface->record(command_ctx.command_buffer);
		}

		VkImageMemoryBarrier swapchain_barrier{};
		swapchain_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		swapchain_barrier.oldLayout           = swapchain_ctx.image_initialized[image_index] ?
		                                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
		                                            VK_IMAGE_LAYOUT_UNDEFINED;
		swapchain_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		swapchain_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		swapchain_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		swapchain_barrier.image               = swapchain_ctx.images[image_index];
		swapchain_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		swapchain_barrier.srcAccessMask       = 0;
		swapchain_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;

		vkCmdPipelineBarrier(command_ctx.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
		                     &swapchain_barrier);

		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.srcOffsets[0]  = {0, 0, 0};
		blit.srcOffsets[1]  = {(int32_t) m_window->width(), (int32_t) m_window->height(), 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.dstOffsets[0]  = {0, 0, 0};
		blit.dstOffsets[1]  = {static_cast<int32_t>(swapchain_ctx.swapchain.extent.width),
		                       static_cast<int32_t>(swapchain_ctx.swapchain.extent.height), 1};

		vkCmdBlitImage(command_ctx.command_buffer, m_water_surface->outputImage(),
		               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_ctx.images[image_index],
		               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

		VkImageMemoryBarrier overlay_barrier{};
		overlay_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		overlay_barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		overlay_barrier.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		overlay_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		overlay_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		overlay_barrier.image               = swapchain_ctx.images[image_index];
		overlay_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		overlay_barrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		overlay_barrier.dstAccessMask =
		    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		vkCmdPipelineBarrier(command_ctx.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0,
		                     nullptr, 1, &overlay_barrier);

		VkRenderPassBeginInfo render_pass_info{};
		render_pass_info.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_info.renderPass  = overlay_ctx.render_pass;
		render_pass_info.framebuffer = overlay_ctx.framebuffers[image_index];
		render_pass_info.renderArea  = {{0, 0}, swapchain_ctx.extent};

		vkCmdBeginRenderPass(command_ctx.command_buffer, &render_pass_info,
		                     VK_SUBPASS_CONTENTS_INLINE);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_ctx.command_buffer);
		vkCmdEndRenderPass(command_ctx.command_buffer);

		check_vk_result(vkEndCommandBuffer(command_ctx.command_buffer));

		const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

		VkSubmitInfo submit_info{};
		submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount   = 1;
		submit_info.pWaitSemaphores      = &sync_ctx.image_available;
		submit_info.pWaitDstStageMask    = &wait_stage;
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &command_ctx.command_buffer;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &sync_ctx.render_finished;

		check_vk_result(
		    vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &submit_info, sync_ctx.in_flight));

		VkPresentInfoKHR present_info{};
		present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores    = &sync_ctx.render_finished;
		present_info.swapchainCount     = 1;
		present_info.pSwapchains        = &swapchain_ctx.swapchain.swapchain;
		present_info.pImageIndices      = &image_index;

		check_vk_result(vkQueuePresentKHR(vulkan_ctx.graphics_queue, &present_info));
		swapchain_ctx.image_initialized[image_index] = true;
	}
}
