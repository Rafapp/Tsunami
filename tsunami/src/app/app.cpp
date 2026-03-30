#include <iostream>

#include "volk.h"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "VkBootstrap.h"
#include "slang.h"
#include "vk_mem_alloc.h"

#include "tsunami/app/app.h"

struct VulkanContext {
	vkb::Instance       instance;
	vkb::PhysicalDevice phys_device;
	vkb::Device         log_device;
	VkDevice            device;
	VkSurfaceKHR        surface;
	VkQueue             graphics_queue;
	uint32_t            graphics_queue_family;
} vulkan_ctx;

struct SwapchainContext {
	vkb::Swapchain           swapchain;
	std::vector<VkImage>     images;
	std::vector<VkImageView> image_views;
	VkFormat                 image_format;
	VkExtent2D               extent;
} swapchain_ctx;

struct RenderTargetContext {
	VmaAllocator          allocator;
	VkImage               storage_image;
	VmaAllocation         storage_image_alloc;
	VkImageView           storage_image_view;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorPool      descriptor_pool;
	VkDescriptorSet       descriptor_set;
} render_target_ctx;

struct ComputePipelineContext {
	VkPipelineLayout pipeline_layout;
	VkPipeline       pipeline;
} compute_ctx;

struct CommandContext {
	VkCommandPool   command_pool;
	VkCommandBuffer command_buffer;
} command_ctx;

struct SyncContext {
	VkSemaphore image_available;
	VkSemaphore render_finished;
	VkFence     in_flight;
} sync_ctx;

static std::vector<uint32_t> compile_slang_shader(const std::string& path,
                                                  const std::string& entry_point) {
	SlangSession*        session = spCreateSession(nullptr);
	SlangCompileRequest* request = spCreateCompileRequest(session);

	// 1. Set target to SPIR-V
	int target_idx = spAddCodeGenTarget(request, SLANG_SPIRV);
	spSetTargetProfile(request, target_idx, spFindProfile(session, "spirv_1_3"));

	// 2. Add the shader file
	int unit_idx = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
	spAddTranslationUnitSourceFile(request, unit_idx, path.c_str());

	// 3. Add entry point
	spAddEntryPoint(request, unit_idx, entry_point.c_str(), SLANG_STAGE_COMPUTE);

	// 4. Compile
	if (spCompile(request) != SLANG_OK) {
		const char* diagnostics = spGetDiagnosticOutput(request);
		spDestroyCompileRequest(request);
		spDestroySession(session);
		throw std::runtime_error(std::string("slang compile error:\n") + diagnostics);
	}

	// 5. Extract SPIR-V
	size_t      spirv_size = 0;
	const void* spirv_data = spGetEntryPointCode(request, 0, &spirv_size);

	std::vector<uint32_t> spirv(spirv_size / sizeof(uint32_t));
	memcpy(spirv.data(), spirv_data, spirv_size);

	spDestroyCompileRequest(request);
	spDestroySession(session);

	return spirv;
}

App::App() {
	// ========================================
	// === I. Load vulkan function pointers ===
	// ========================================

	// TODO: Logging class and functionality

	if (volkInitialize() != VK_SUCCESS) {
		throw std::runtime_error("failed to initialize volk");
	}
	std::cout << "[INFO] Initialized volk\n";

	// =========================
	// === II. Create window ===
	// =========================
	m_window = std::make_unique<core::Window>(
	    core::WindowConfig{.width = 1280, .height = 720, .title = "tsunami 🌊"});
	std::cout << "[INFO] Created window\n";

	// ==================================
	// === III. Create Vulkan context ===
	// ==================================

	// 1. Create Vulkan instance
	vkb::InstanceBuilder builder;
	auto                 inst_ret = builder.set_app_name("tsunami")
	                                    .request_validation_layers()
	                                    .use_default_debug_messenger()
	                                    .require_api_version(1, 3, 0)
	                                    .build();
	if (!inst_ret)
		throw std::runtime_error("failed to create Vulkan instance");
	vulkan_ctx.instance = inst_ret.value();
	volkLoadInstance(vulkan_ctx.instance.instance);
	std::cout << "[INFO] Created Vulkan instance\n";

	// 2. Create a surface
	vulkan_ctx.surface = VK_NULL_HANDLE;
	if (glfwCreateWindowSurface(vulkan_ctx.instance, m_window->handle(), nullptr,
	                            &vulkan_ctx.surface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface");
	}
	std::cout << "[INFO] Created window surface\n";

	// 5. Select physical device (GPU)
	auto phys_dev_ret = vkb::PhysicalDeviceSelector(vulkan_ctx.instance)
	                        .set_surface(vulkan_ctx.surface)
	                        .set_minimum_version(1, 3)
	                        .select();
	if (!phys_dev_ret)
		throw std::runtime_error("failed to select physical device");
	vulkan_ctx.phys_device = phys_dev_ret.value();
	std::cout << "[INFO] Selected physical device\n";

	// 6. Create logical device and load with volk
	vkb::DeviceBuilder device_builder{vulkan_ctx.phys_device};
	auto               dev_ret = device_builder.build();
	if (!dev_ret)
		throw std::runtime_error("failed to create logical device");
	vulkan_ctx.log_device = dev_ret.value();
	vulkan_ctx.device     = vulkan_ctx.log_device.device;
	volkLoadDevice(vulkan_ctx.device);
	std::cout << "[INFO] Created logical device\n";

	// 7. Create graphics queue, get queue and index
	auto graphics_queue_ret = vulkan_ctx.log_device.get_queue(vkb::QueueType::graphics);
	if (!graphics_queue_ret)
		throw std::runtime_error("failed to get graphics queue");
	vulkan_ctx.graphics_queue = graphics_queue_ret.value();
	std::cout << "[INFO] Created graphics queue\n";

	auto family_ret = vulkan_ctx.log_device.get_queue_index(vkb::QueueType::graphics);
	if (!family_ret)
		throw std::runtime_error("failed to get graphics queue family");
	vulkan_ctx.graphics_queue_family = family_ret.value();

	// ==============================================
	// === IV. Initialize VMA to write to VkImage ===
	// ==============================================
	VmaAllocatorCreateInfo vma_info{};
	vma_info.instance         = vulkan_ctx.instance.instance;
	vma_info.physicalDevice   = vulkan_ctx.phys_device.physical_device;
	vma_info.device           = vulkan_ctx.device;
	vma_info.vulkanApiVersion = VK_API_VERSION_1_3;

	VmaVulkanFunctions vma_vulkan_funcs{};
	vma_vulkan_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	vma_vulkan_funcs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
	vma_info.pVulkanFunctions              = &vma_vulkan_funcs;

	VmaAllocator allocator;
	if (vmaCreateAllocator(&vma_info, &allocator) != VK_SUCCESS) {
		throw std::runtime_error("failed to create VMA allocator");
	}
	std::cout << "[INFO] Created VMA allocator\n";

	// ====================================
	// === V. Create swapchain context ===
	// ====================================

	// 1. Create swapchain
	vkb::SwapchainBuilder swapchain_builder{vulkan_ctx.log_device};
	auto                  swap_ret =
	    swapchain_builder
	        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
	        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
	        .set_desired_extent(m_window->width(), m_window->height())
	        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	        .build();
	if (!swap_ret)
		throw std::runtime_error("failed to create swapchain");
	swapchain_ctx.swapchain    = swap_ret.value();
	swapchain_ctx.image_format = swapchain_ctx.swapchain.image_format;
	std::cout << "[INFO] Created swapchain (format: " << swapchain_ctx.image_format << ")\n";

	// 2. Create swapchain images
	auto images_ret = swapchain_ctx.swapchain.get_images();
	if (!images_ret)
		throw std::runtime_error("failed to get swapchain images");
	swapchain_ctx.images = images_ret.value();
	std::cout << "[INFO] Acquired " << swapchain_ctx.images.size() << " swapchain images\n";

	// 3. Create swapchain image views
	auto image_views_ret = swapchain_ctx.swapchain.get_image_views();
	if (!image_views_ret)
		throw std::runtime_error("failed to get swapchain image views");
	swapchain_ctx.image_views = image_views_ret.value();
	std::cout << "[INFO] Created swapchain image views\n";

	// ========================================
	// === VI. Create Render Target Context ===
	// ========================================

	// 1. Define storage image info
	VkImageCreateInfo image_info{};
	image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType     = VK_IMAGE_TYPE_2D;
	image_info.format        = VK_FORMAT_R8G8B8A8_UNORM;
	image_info.extent        = {(uint32_t) m_window->width(), (uint32_t) m_window->height(), 1};
	image_info.mipLevels     = 1;
	image_info.arrayLayers   = 1;
	image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
	image_info.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	// 2. Create storage image & allocate memory
	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	if (vmaCreateImage(allocator, &image_info, &alloc_info, &render_target_ctx.storage_image,
	                   &render_target_ctx.storage_image_alloc, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("failed to create storage image");
	}
	std::cout << "[INFO] Created storage image (resolution: " << image_info.extent.width << "x"
	          << image_info.extent.height << ")\n";

	// 3. Create image view
	VkImageViewCreateInfo view_info{};
	view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image                           = render_target_ctx.storage_image;
	view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format                          = VK_FORMAT_R8G8B8A8_UNORM;
	view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	view_info.subresourceRange.baseMipLevel   = 0;
	view_info.subresourceRange.levelCount     = 1;
	view_info.subresourceRange.baseArrayLayer = 0;
	view_info.subresourceRange.layerCount     = 1;

	if (vkCreateImageView(vulkan_ctx.device, &view_info, nullptr,
	                      &render_target_ctx.storage_image_view) != VK_SUCCESS) {
		throw std::runtime_error("failed to create storage image view");
	}
	std::cout << "[INFO] Created storage image view\n";

	// 4. Create descriptor set layout
	VkDescriptorSetLayoutBinding storage_image_binding{};
	storage_image_binding.binding         = 0;
	storage_image_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	storage_image_binding.descriptorCount = 1;
	storage_image_binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo dsl_info{};
	dsl_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dsl_info.bindingCount = 1;
	dsl_info.pBindings    = &storage_image_binding;

	if (vkCreateDescriptorSetLayout(vulkan_ctx.device, &dsl_info, nullptr,
	                                &render_target_ctx.descriptor_set_layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout");
	}
	std::cout << "[INFO] Created descriptor set layout\n";

	// 5. Create descriptor pool
	VkDescriptorPoolSize pool_size{};
	pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	pool_size.descriptorCount = 1;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;

	if (vkCreateDescriptorPool(vulkan_ctx.device, &pool_info, nullptr,
	                           &render_target_ctx.descriptor_pool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool");
	}
	std::cout << "[INFO] Created descriptor pool\n";

	// 6. Allocate descriptor set
	VkDescriptorSetAllocateInfo descriptor_alloc_info{};
	descriptor_alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptor_alloc_info.descriptorPool     = render_target_ctx.descriptor_pool;
	descriptor_alloc_info.descriptorSetCount = 1;
	descriptor_alloc_info.pSetLayouts        = &render_target_ctx.descriptor_set_layout;

	if (vkAllocateDescriptorSets(vulkan_ctx.device, &descriptor_alloc_info,
	                             &render_target_ctx.descriptor_set) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor set");
	}
	std::cout << "[INFO] Allocated descriptor set\n";

	// 7. Update descriptor set with image info
	VkDescriptorImageInfo descriptor_image_info{};
	descriptor_image_info.imageView   = render_target_ctx.storage_image_view;
	descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet write{};
	write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet          = render_target_ctx.descriptor_set;
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	write.pImageInfo      = &descriptor_image_info;

	vkUpdateDescriptorSets(vulkan_ctx.device, 1, &write, 0, nullptr);
	std::cout << "[INFO] Updated descriptor set with storage image\n";

	// ============================================
	// === VII. Create Compute Pipeline Context ===
	// ============================================

	// 1. Compile shader
	// TODO: Extract shading system/caching to "shading.h/cpp"
	auto spirv = compile_slang_shader("shaders/random_color.slang", "main");
	std::cout << "[INFO] Compiled compute shader to SPIR-V (" << (spirv.size() * sizeof(uint32_t))
	          << " bytes)\n";

	// 2. Create shader module
	VkShaderModuleCreateInfo module_info{};
	module_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	module_info.codeSize = spirv.size() * sizeof(uint32_t);
	module_info.pCode    = spirv.data();

	VkShaderModule shader_module;
	if (vkCreateShaderModule(vulkan_ctx.device, &module_info, nullptr, &shader_module) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module");
	}
	std::cout << "[INFO] Created shader module\n";

	// 3. Pipeline layout, references our descriptor set layout
	VkPushConstantRange push_constant_range{};
	push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	push_constant_range.offset     = 0;
	push_constant_range.size       = sizeof(uint32_t);

	VkPipelineLayoutCreateInfo pipeline_layout_info{};
	pipeline_layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &render_target_ctx.descriptor_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constant_range;

	if (vkCreatePipelineLayout(vulkan_ctx.device, &pipeline_layout_info, nullptr,
	                           &compute_ctx.pipeline_layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout");
	}
	std::cout << "[INFO] Created pipeline layout\n";

	// 4. Create compute pipeline
	VkPipelineShaderStageCreateInfo stage_info{};
	stage_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage_info.module = shader_module;
	stage_info.pName  = "main";

	VkComputePipelineCreateInfo pipeline_info{};
	pipeline_info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeline_info.stage  = stage_info;
	pipeline_info.layout = compute_ctx.pipeline_layout;

	if (vkCreateComputePipelines(vulkan_ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
	                             &compute_ctx.pipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline");
	}
	std::cout << "[INFO] Created compute pipeline\n";

	// 5. Shader module baked into pipeline, safe to destroy now
	vkDestroyShaderModule(vulkan_ctx.device, shader_module, nullptr);

	// ==========================================
	// === VIII. Create command pool & buffer ===
	// ==========================================

	// 1. Create command pool, allocator for command buffers, tied to a queue family
	VkCommandPoolCreateInfo cmd_pool_info{};
	cmd_pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmd_pool_info.queueFamilyIndex = vulkan_ctx.graphics_queue_family;
	cmd_pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (vkCreateCommandPool(vulkan_ctx.device, &cmd_pool_info, nullptr,
	                        &command_ctx.command_pool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool");
	}
	std::cout << "[INFO] Created command pool\n";

	// 2. Create command buffer, allocated from the pool (one is enough for now)
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

	// ========================
	// === IX. Sync Context ===
	// ========================

	VkSemaphoreCreateInfo semaphore_info{};
	semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_info{};
	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	// 1. Create semaphore
	if (vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr, &sync_ctx.image_available) !=
	        VK_SUCCESS ||
	    vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr, &sync_ctx.render_finished) !=
	        VK_SUCCESS ||
	    // 2. Create fence

	    vkCreateFence(vulkan_ctx.device, &fence_info, nullptr, &sync_ctx.in_flight) != VK_SUCCESS) {
		throw std::runtime_error("failed to create sync objects");
	}

	std::cout << "[INFO] Created sync objects (image_available, render_finished, in_flight)\n";
}

void App::run() {
	MainLoop();
}

void App::MainLoop() {
	uint32_t frame_number = 0;
	while (!m_window->shouldClose()) {
		m_window->pollEvents();
		glfwPollEvents();

		// 1. Wait for previous frame to finish
		vkWaitForFences(vulkan_ctx.device, 1, &sync_ctx.in_flight, VK_TRUE, UINT64_MAX);
		vkResetFences(vulkan_ctx.device, 1, &sync_ctx.in_flight);

		// 2. Acquire next swapchain image
		uint32_t image_index;
		vkAcquireNextImageKHR(vulkan_ctx.device, swapchain_ctx.swapchain.swapchain, UINT64_MAX,
		                      sync_ctx.image_available, VK_NULL_HANDLE, &image_index);

		// 3. Record command buffer
		vkResetCommandBuffer(command_ctx.command_buffer, 0);

		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(command_ctx.command_buffer, &begin_info);

		// 3a. Transition storage image to GENERAL so compute can write to it
		VkImageMemoryBarrier storage_barrier{};
		storage_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		storage_barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
		storage_barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
		storage_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		storage_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		storage_barrier.image               = render_target_ctx.storage_image;
		storage_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		storage_barrier.srcAccessMask       = 0;
		storage_barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

		vkCmdPipelineBarrier(command_ctx.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
		                     &storage_barrier);

		// 3b. Bind compute pipeline and descriptor set
		vkCmdBindPipeline(command_ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		                  compute_ctx.pipeline);
		vkCmdBindDescriptorSets(command_ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        compute_ctx.pipeline_layout, 0, 1,
		                        &render_target_ctx.descriptor_set, 0, nullptr);

		// 3c. Dispatch — one thread per pixel, groups of 16x16
		uint32_t group_x = (m_window->width() + 15) / 16;
		uint32_t group_y = (m_window->height() + 15) / 16;
		vkCmdPushConstants(command_ctx.command_buffer, compute_ctx.pipeline_layout,
		                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &frame_number);
		vkCmdDispatch(command_ctx.command_buffer, group_x, group_y, 1);
		frame_number++;

		// 3d. Transition storage image to TRANSFER_SRC for blit
		VkImageMemoryBarrier transfer_barrier{};
		transfer_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		transfer_barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
		transfer_barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		transfer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		transfer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		transfer_barrier.image               = render_target_ctx.storage_image;
		transfer_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		transfer_barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
		transfer_barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(command_ctx.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
		                     &transfer_barrier);

		// 3e. Transition swapchain image to TRANSFER_DST for blit
		VkImageMemoryBarrier swapchain_barrier{};
		swapchain_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		swapchain_barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
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

		// 3f. Blit storage image → swapchain image
		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.srcOffsets[0]  = {0, 0, 0};
		blit.srcOffsets[1]  = {(int32_t) m_window->width(), (int32_t) m_window->height(), 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.dstOffsets[0]  = {0, 0, 0};
		blit.dstOffsets[1]  = {(int32_t) swapchain_ctx.swapchain.extent.width,
		                       (int32_t) swapchain_ctx.swapchain.extent.height, 1};

		vkCmdBlitImage(command_ctx.command_buffer, render_target_ctx.storage_image,
		               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_ctx.images[image_index],
		               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

		// 3g. Transition swapchain image to PRESENT_SRC
		VkImageMemoryBarrier present_barrier{};
		present_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		present_barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		present_barrier.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		present_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		present_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		present_barrier.image               = swapchain_ctx.images[image_index];
		present_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		present_barrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		present_barrier.dstAccessMask       = 0;

		vkCmdPipelineBarrier(command_ctx.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1,
		                     &present_barrier);

		vkEndCommandBuffer(command_ctx.command_buffer);

		// 4. Submit to queue
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		VkSubmitInfo submit_info{};
		submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount   = 1;
		submit_info.pWaitSemaphores      = &sync_ctx.image_available;
		submit_info.pWaitDstStageMask    = &wait_stage;
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &command_ctx.command_buffer;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &sync_ctx.render_finished;

		vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &submit_info, sync_ctx.in_flight);

		// 5. Present
		VkPresentInfoKHR present_info{};
		present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores    = &sync_ctx.render_finished;
		present_info.swapchainCount     = 1;
		present_info.pSwapchains        = &swapchain_ctx.swapchain.swapchain;
		present_info.pImageIndices      = &image_index;

		vkQueuePresentKHR(vulkan_ctx.graphics_queue, &present_info);
	}

	// wait for GPU to finish before cleanup
	vkDeviceWaitIdle(vulkan_ctx.device);
}