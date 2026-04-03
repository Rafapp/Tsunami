#include <array>
#include <iostream>

#define VOLK_IMPLEMENTATION
#include "volk.h"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "VkBootstrap.h"
#include "slang.h"
#include "tsunami/app/app.h"
#include "vk_mem_alloc.h"

#include "tsunami/app/app.h"

struct SceneContext {
	VkBuffer      camera_buffer   = VK_NULL_HANDLE;
	VmaAllocation camera_alloc    = VK_NULL_HANDLE;
	VkBuffer      shapes_buffer   = VK_NULL_HANDLE;
	VmaAllocation shapes_alloc    = VK_NULL_HANDLE;
	VkBuffer      material_buffer = VK_NULL_HANDLE;
	VmaAllocation material_alloc  = VK_NULL_HANDLE;
	uint32_t      shape_count     = 0;
	uint32_t      material_count  = 0;
} scene_ctx;

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
	VmaAllocator allocator;

	VkImage       storage_image;
	VmaAllocation storage_image_alloc;
	VkImageView   storage_image_view;

	// Added accumulation buffer for progressive rendering (not used in the shader yet, but set up
	// for future use)
	VkImage       accum_image;
	VmaAllocation accum_image_alloc;
	VkImageView   accum_image_view;

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
	VkSemaphore              image_available;
	std::vector<VkSemaphore> render_finished;
	VkFence                  in_flight;
} sync_ctx;

static std::vector<uint32_t> compile_slang_shader(const std::string& path,
                                                  const std::string& entry_point) {
	SlangSession*        session = spCreateSession(nullptr);
	SlangCompileRequest* request = spCreateCompileRequest(session);

	int target_idx = spAddCodeGenTarget(request, SLANG_SPIRV);
	spSetTargetProfile(request, target_idx, spFindProfile(session, "spirv_1_3"));

	int unit_idx = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
	spAddTranslationUnitSourceFile(request, unit_idx, path.c_str());

	spAddEntryPoint(request, unit_idx, entry_point.c_str(), SLANG_STAGE_COMPUTE);

	SlangResult result = spCompile(request);

	// Always print — catches warnings even on success
	const char* diagnostics = spGetDiagnosticOutput(request);
	if (diagnostics && diagnostics[0] != '\0') {
		std::cerr << "[SLANG] " << path << ":\n" << diagnostics << "\n";
	}

	if (result != SLANG_OK) {
		spDestroyCompileRequest(request);
		spDestroySession(session);
		throw std::runtime_error("slang compilation failed: " + path);
	}

	size_t      spirv_size = 0;
	const void* spirv_data = spGetEntryPointCode(request, 0, &spirv_size);

	std::vector<uint32_t> spirv(spirv_size / sizeof(uint32_t));
	memcpy(spirv.data(), spirv_data, spirv_size);

	spDestroyCompileRequest(request);
	spDestroySession(session);

	return spirv;
}

App::App() {
	// ======================
	// === 0. Scene setup ===
	// ======================

	// TODO: Parse scene description from file or USD
	m_scene = std::make_unique<Scene>();

	// Camera
	m_scene->m_camera = Camera(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f),
	                           glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 0.1f, 100.0f);

	// Floor (white)
	m_scene->m_shapes.push_back(std::make_unique<Quad>(
	    Transform(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f), glm::vec3(2.0f, 1.0f, 2.0f)),
	    new Lambert(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Ceiling (white)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(180.0f, 0.0f, 0.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           new Lambert(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Back wall (white)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(90.0f, 0.0f, 0.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           new Lambert(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Left wall (red)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 90.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           new Lambert(glm::vec3(0.8f, 0.1f, 0.1f))));

	// Right wall (green)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -90.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           new Lambert(glm::vec3(0.1f, 0.8f, 0.1f))));

	// Area light (emissive quad on ceiling)
	m_scene->m_shapes.push_back(std::make_unique<Quad>(
	    Transform(glm::vec3(0.0f, 0.99f, 0.0f), glm::vec3(180.0f, 0.0f, 0.0f),
	              glm::vec3(2.5f, 1.0f, 2.5f)),
	    new Lambert(glm::vec3(1.0f), glm::vec3(15.0f), 0.25f)));

	// Tall box
	m_scene->m_shapes.push_back(
	    std::make_unique<Box>(Transform(glm::vec3(-0.35f, -0.35f, -0.4f),
	                                    glm::vec3(0.0f, 15.0f, 0.0f), glm::vec3(0.3f, 0.65f, 0.3f)),
	                          new Lambert(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Short box
	m_scene->m_shapes.push_back(std::make_unique<Box>(Transform(glm::vec3(0.35f, -0.65f, -0.2f),
	                                                            glm::vec3(0.0f, -15.0f, 0.0f),
	                                                            glm::vec3(0.3f, 0.35f, 0.3f)),
	                                                  new Lambert(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Teapot (mesh)
	m_scene->m_meshes.push_back(std::make_unique<Mesh>(
	    "resources/meshes/teapot.obj",
	    Transform(glm::vec3(0.0f, -0.5f, 0.5f), glm::vec3(0.0f, 45.0f, 0.0f), glm::vec3(0.5f)),
	    new Lambert(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Pack scene data for GPU
	GPUCamera                gpu_camera = m_scene->m_camera.pack();
	std::vector<GPUShape>    gpu_shapes;
	std::vector<GPUMaterial> gpu_materials;

	for (auto& shape : m_scene->m_shapes) {
		int matIndex = (int) gpu_materials.size();
		gpu_materials.push_back(shape->m_material->pack());
		gpu_shapes.push_back(shape->pack(matIndex));
	}

	scene_ctx.shape_count    = (uint32_t) gpu_shapes.size();
	scene_ctx.material_count = (uint32_t) gpu_materials.size();

	VkDeviceSize camera_size   = sizeof(GPUCamera);
	VkDeviceSize shapes_size   = sizeof(GPUShape) * gpu_shapes.size();
	VkDeviceSize material_size = sizeof(GPUMaterial) * gpu_materials.size();

	// ========================================
	// === I. Load vulkan function pointers ===
	// ========================================

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

	vulkan_ctx.surface = VK_NULL_HANDLE;
	if (glfwCreateWindowSurface(vulkan_ctx.instance, m_window->handle(), nullptr,
	                            &vulkan_ctx.surface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface");
	}
	std::cout << "[INFO] Created window surface\n";

	std::vector<const char*> device_extensions = {
	    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	    VK_KHR_RAY_QUERY_EXTENSION_NAME,
	    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	};

	auto phys_dev_ret = vkb::PhysicalDeviceSelector(vulkan_ctx.instance)
	                        .set_surface(vulkan_ctx.surface)
	                        .set_minimum_version(1, 3)
	                        .add_required_extensions(device_extensions)
	                        .select();

	// Fail graciously if there is no hardware ray tracing support
	// TODO: Create compute-only version of Tsunami based in current foundation
	if (!phys_dev_ret) {
		std::cout << "[ERROR] Device selection failed. Checking devices:\n";

		uint32_t count = 0;
		vkEnumeratePhysicalDevices(vulkan_ctx.instance, &count, nullptr);
		std::vector<VkPhysicalDevice> devs(count);
		vkEnumeratePhysicalDevices(vulkan_ctx.instance, &count, devs.data());

		for (auto d : devs) {
			VkPhysicalDeviceProperties p{};
			vkGetPhysicalDeviceProperties(d, &p);

			VkPhysicalDeviceRayQueryFeaturesKHR rq{
			    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
			VkPhysicalDeviceAccelerationStructureFeaturesKHR as{
			    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
			VkPhysicalDeviceBufferDeviceAddressFeatures bda{
			    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};

			rq.pNext = &as;
			as.pNext = &bda;

			VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
			f.pNext = &rq;
			vkGetPhysicalDeviceFeatures2(d, &f);

			std::cout << "  " << p.deviceName << " | rayQuery=" << rq.rayQuery
			          << " accelStruct=" << as.accelerationStructure
			          << " bda=" << bda.bufferDeviceAddress << "\n";
		}

		throw std::runtime_error("failed to select physical device");
	}

	vulkan_ctx.phys_device = phys_dev_ret.value();

	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(vulkan_ctx.phys_device.physical_device, &props);
	std::cout << "[INFO] Selected physical device: " << props.deviceName << "\n";

	VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
	as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	as_features.accelerationStructure = VK_TRUE;

	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{};
	ray_query_features.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
	ray_query_features.rayQuery = VK_TRUE;

	VkPhysicalDeviceBufferDeviceAddressFeatures bda_features{};
	bda_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
	bda_features.bufferDeviceAddress = VK_TRUE;

	vkb::DeviceBuilder device_builder{vulkan_ctx.phys_device};
	auto               dev_ret = device_builder.add_pNext(&as_features)
	                                 .add_pNext(&ray_query_features)
	                                 .add_pNext(&bda_features)
	                                 .build();
	if (!dev_ret)
		throw std::runtime_error("failed to create logical device");
	vulkan_ctx.log_device = dev_ret.value();
	vulkan_ctx.device     = vulkan_ctx.log_device.device;
	volkLoadDevice(vulkan_ctx.device);
	std::cout << "[INFO] Created logical device\n";

	auto graphics_queue_ret = vulkan_ctx.log_device.get_queue(vkb::QueueType::graphics);
	if (!graphics_queue_ret)
		throw std::runtime_error("failed to get graphics queue");
	vulkan_ctx.graphics_queue = graphics_queue_ret.value();
	std::cout << "[INFO] Created graphics queue\n";

	auto family_ret = vulkan_ctx.log_device.get_queue_index(vkb::QueueType::graphics);
	if (!family_ret)
		throw std::runtime_error("failed to get graphics queue family");
	vulkan_ctx.graphics_queue_family = family_ret.value();

	// ==========================
	// === IV. Initialize VMA ===
	// ==========================
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
	render_target_ctx.allocator = allocator;

	// ===================================
	// === V. Create swapchain context ===
	// ===================================
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

	auto images_ret = swapchain_ctx.swapchain.get_images();
	if (!images_ret)
		throw std::runtime_error("failed to get swapchain images");
	swapchain_ctx.images = images_ret.value();
	std::cout << "[INFO] Acquired " << swapchain_ctx.images.size() << " swapchain images\n";

	auto image_views_ret = swapchain_ctx.swapchain.get_image_views();
	if (!image_views_ret)
		throw std::runtime_error("failed to get swapchain image views");
	swapchain_ctx.image_views = image_views_ret.value();
	std::cout << "[INFO] Created swapchain image views\n";

	// ========================================
	// === VI. Create Render Target Context ===
	// ========================================

	// 1. Storage image
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

	VmaAllocationCreateInfo img_alloc_info{};
	img_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	if (vmaCreateImage(allocator, &image_info, &img_alloc_info, &render_target_ctx.storage_image,
	                   &render_target_ctx.storage_image_alloc, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("failed to create storage image");
	}
	std::cout << "[INFO] Created storage image (" << image_info.extent.width << "x"
	          << image_info.extent.height << ")\n";

	// 2. Accum image
	VkImageCreateInfo accum_info = image_info;
	accum_info.usage             = VK_IMAGE_USAGE_STORAGE_BIT;

	if (vmaCreateImage(allocator, &accum_info, &img_alloc_info, &render_target_ctx.accum_image,
	                   &render_target_ctx.accum_image_alloc, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("failed to create accum image");
	}

	VkImageViewCreateInfo accum_view_info{};
	accum_view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	accum_view_info.image                           = render_target_ctx.accum_image;
	accum_view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	accum_view_info.format                          = VK_FORMAT_R8G8B8A8_UNORM;
	accum_view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	accum_view_info.subresourceRange.baseMipLevel   = 0;
	accum_view_info.subresourceRange.levelCount     = 1;
	accum_view_info.subresourceRange.baseArrayLayer = 0;
	accum_view_info.subresourceRange.layerCount     = 1;

	if (vkCreateImageView(vulkan_ctx.device, &accum_view_info, nullptr,
	                      &render_target_ctx.accum_image_view) != VK_SUCCESS) {
		throw std::runtime_error("failed to create accum image view");
	}
	std::cout << "[INFO] Created accum image and view\n";

	// 3. Storage image view
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

	// =================================
	// === VI.5 Create Scene Buffers ===
	// =================================

	auto createAndUploadBuffer = [&](VkDeviceSize size, const void* data, VkBuffer& buffer,
	                                 VmaAllocation& alloc) {
		VkBufferCreateInfo buf_info{};
		buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buf_info.size  = size;
		buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		VmaAllocationCreateInfo buf_alloc_info{};
		buf_alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
		buf_alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VmaAllocationInfo info;
		if (vmaCreateBuffer(allocator, &buf_info, &buf_alloc_info, &buffer, &alloc, &info) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create scene buffer");
		memcpy(info.pMappedData, data, size);
	};

	createAndUploadBuffer(camera_size, &gpu_camera, scene_ctx.camera_buffer,
	                      scene_ctx.camera_alloc);
	createAndUploadBuffer(shapes_size, gpu_shapes.data(), scene_ctx.shapes_buffer,
	                      scene_ctx.shapes_alloc);
	createAndUploadBuffer(material_size, gpu_materials.data(), scene_ctx.material_buffer,
	                      scene_ctx.material_alloc);
	std::cout << "[INFO] Uploaded scene buffers (shapes: " << scene_ctx.shape_count
	          << ", materials: " << scene_ctx.material_count << ")\n";

	// 4. Descriptor set layout (bindings 0-4)
	auto makeImageBinding = [](uint32_t binding) {
		VkDescriptorSetLayoutBinding b{};
		b.binding         = binding;
		b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		b.descriptorCount = 1;
		b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
		return b;
	};
	auto makeBufferBinding = [](uint32_t binding) {
		VkDescriptorSetLayoutBinding b{};
		b.binding         = binding;
		b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		b.descriptorCount = 1;
		b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
		return b;
	};

	std::array<VkDescriptorSetLayoutBinding, 5> bindings = {
	    makeImageBinding(0),         // output image
	    makeImageBinding(1),         // accum image
	    makeBufferBinding(2),        // camera
	    makeBufferBinding(3),        // shapes
	    makeBufferBinding(4),        // materials
	};

	VkDescriptorSetLayoutCreateInfo dsl_info{};
	dsl_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dsl_info.bindingCount = (uint32_t) bindings.size();
	dsl_info.pBindings    = bindings.data();

	if (vkCreateDescriptorSetLayout(vulkan_ctx.device, &dsl_info, nullptr,
	                                &render_target_ctx.descriptor_set_layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout");
	}
	std::cout << "[INFO] Created descriptor set layout\n";

	// 5. Descriptor pool
	std::array<VkDescriptorPoolSize, 2> pool_sizes{};
	pool_sizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	pool_sizes[0].descriptorCount = 2;
	pool_sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pool_sizes[1].descriptorCount = 3;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = (uint32_t) pool_sizes.size();
	pool_info.pPoolSizes    = pool_sizes.data();

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

	// 7. Update descriptor set
	VkDescriptorImageInfo descriptor_image_info{};
	descriptor_image_info.imageView   = render_target_ctx.storage_image_view;
	descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo descriptor_accum_image_info{};
	descriptor_accum_image_info.imageView   = render_target_ctx.accum_image_view;
	descriptor_accum_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorBufferInfo camera_buf_info{};
	camera_buf_info.buffer = scene_ctx.camera_buffer;
	camera_buf_info.offset = 0;
	camera_buf_info.range  = camera_size;

	VkDescriptorBufferInfo shapes_buf_info{};
	shapes_buf_info.buffer = scene_ctx.shapes_buffer;
	shapes_buf_info.offset = 0;
	shapes_buf_info.range  = shapes_size;

	VkDescriptorBufferInfo material_buf_info{};
	material_buf_info.buffer = scene_ctx.material_buffer;
	material_buf_info.offset = 0;
	material_buf_info.range  = material_size;

	std::array<VkWriteDescriptorSet, 5> writes{};

	writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet          = render_target_ctx.descriptor_set;
	writes[0].dstBinding      = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[0].pImageInfo      = &descriptor_image_info;

	writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet          = render_target_ctx.descriptor_set;
	writes[1].dstBinding      = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[1].pImageInfo      = &descriptor_accum_image_info;

	writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet          = render_target_ctx.descriptor_set;
	writes[2].dstBinding      = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[2].pBufferInfo     = &camera_buf_info;

	writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[3].dstSet          = render_target_ctx.descriptor_set;
	writes[3].dstBinding      = 3;
	writes[3].descriptorCount = 1;
	writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[3].pBufferInfo     = &shapes_buf_info;

	writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[4].dstSet          = render_target_ctx.descriptor_set;
	writes[4].dstBinding      = 4;
	writes[4].descriptorCount = 1;
	writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[4].pBufferInfo     = &material_buf_info;

	vkUpdateDescriptorSets(vulkan_ctx.device, (uint32_t) writes.size(), writes.data(), 0, nullptr);
	std::cout << "[INFO] Updated descriptor sets\n";

	// ============================================
	// === VII. Create Compute Pipeline Context ===
	// ============================================

	auto spirv = compile_slang_shader("shaders/cursedPathTracingAlgo.slang", "main");
	std::cout << "[INFO] Compiled compute shader to SPIR-V (" << (spirv.size() * sizeof(uint32_t))
	          << " bytes)\n";

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

	// Push constants: frame, shape_count, material_count
	VkPushConstantRange push_constant_range{};
	push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	push_constant_range.offset     = 0;
	push_constant_range.size       = sizeof(uint32_t) * 3;

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

	vkDestroyShaderModule(vulkan_ctx.device, shader_module, nullptr);

	// ==========================================
	// === VIII. Create command pool & buffer ===
	// ==========================================

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

	// === One-time image transitions ===
	VkCommandBufferAllocateInfo one_time_alloc{};
	one_time_alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	one_time_alloc.commandPool        = command_ctx.command_pool;
	one_time_alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	one_time_alloc.commandBufferCount = 1;

	VkCommandBuffer one_time_cmd;
	vkAllocateCommandBuffers(vulkan_ctx.device, &one_time_alloc, &one_time_cmd);

	VkCommandBufferBeginInfo one_time_begin{};
	one_time_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	one_time_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(one_time_cmd, &one_time_begin);

	VkImageMemoryBarrier accum_init_barrier{};
	accum_init_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	accum_init_barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
	accum_init_barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	accum_init_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	accum_init_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	accum_init_barrier.image               = render_target_ctx.accum_image;
	accum_init_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	accum_init_barrier.srcAccessMask       = 0;
	accum_init_barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

	vkCmdPipelineBarrier(one_time_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
	                     &accum_init_barrier);

	vkEndCommandBuffer(one_time_cmd);

	VkSubmitInfo one_time_submit{};
	one_time_submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	one_time_submit.commandBufferCount = 1;
	one_time_submit.pCommandBuffers    = &one_time_cmd;
	vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &one_time_submit, VK_NULL_HANDLE);
	vkQueueWaitIdle(vulkan_ctx.graphics_queue);
	vkFreeCommandBuffers(vulkan_ctx.device, command_ctx.command_pool, 1, &one_time_cmd);
	std::cout << "[INFO] Transitioned accum image to GENERAL\n";

	// ========================
	// === IX. Sync Context ===
	// ========================

	VkSemaphoreCreateInfo semaphore_info{};
	semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_info{};
	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	sync_ctx.render_finished.resize(swapchain_ctx.images.size());

	if (vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr, &sync_ctx.image_available) !=
	    VK_SUCCESS) {
		throw std::runtime_error("failed to create image available semaphore");
	}

	for (auto& sem : sync_ctx.render_finished) {
		if (vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr, &sem) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render finished semaphore");
		}
	}

	if (vkCreateFence(vulkan_ctx.device, &fence_info, nullptr, &sync_ctx.in_flight) != VK_SUCCESS) {
		throw std::runtime_error("failed to create in-flight fence");
	}

	std::cout << "[INFO] Created sync objects\n";
}

void App::run() {
	MainLoop();
}

void App::MainLoop() {
	uint32_t frame_number = 0;
	while (!m_window->shouldClose()) {
		m_window->pollEvents();
		glfwPollEvents();

		vkWaitForFences(vulkan_ctx.device, 1, &sync_ctx.in_flight, VK_TRUE, UINT64_MAX);
		vkResetFences(vulkan_ctx.device, 1, &sync_ctx.in_flight);

		uint32_t image_index;
		vkAcquireNextImageKHR(vulkan_ctx.device, swapchain_ctx.swapchain.swapchain, UINT64_MAX,
		                      sync_ctx.image_available, VK_NULL_HANDLE, &image_index);

		vkResetCommandBuffer(command_ctx.command_buffer, 0);

		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(command_ctx.command_buffer, &begin_info);

		// Transition storage image to GENERAL
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

		// Bind pipeline and descriptor set
		vkCmdBindPipeline(command_ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		                  compute_ctx.pipeline);
		vkCmdBindDescriptorSets(command_ctx.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        compute_ctx.pipeline_layout, 0, 1,
		                        &render_target_ctx.descriptor_set, 0, nullptr);

		// Push constants: frame, shape_count, material_count
		struct PushConstants {
			uint32_t frame;
			uint32_t shape_count;
			uint32_t material_count;
		};
		PushConstants pc{frame_number, scene_ctx.shape_count, scene_ctx.material_count};
		vkCmdPushConstants(command_ctx.command_buffer, compute_ctx.pipeline_layout,
		                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		uint32_t group_x = (m_window->width() + 15) / 16;
		uint32_t group_y = (m_window->height() + 15) / 16;
		vkCmdDispatch(command_ctx.command_buffer, group_x, group_y, 1);
		frame_number++;

		// Transition storage image to TRANSFER_SRC
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

		// Transition swapchain image to TRANSFER_DST
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

		// Blit storage image → swapchain image
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

		// Transition swapchain image to PRESENT_SRC
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

		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		VkSubmitInfo submit_info{};
		submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount   = 1;
		submit_info.pWaitSemaphores      = &sync_ctx.image_available;
		submit_info.pWaitDstStageMask    = &wait_stage;
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &command_ctx.command_buffer;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &sync_ctx.render_finished[image_index];

		vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &submit_info, sync_ctx.in_flight);

		VkPresentInfoKHR present_info{};
		present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores    = &sync_ctx.render_finished[image_index];
		present_info.swapchainCount     = 1;
		present_info.pSwapchains        = &swapchain_ctx.swapchain.swapchain;
		present_info.pImageIndices      = &image_index;

		vkQueuePresentKHR(vulkan_ctx.graphics_queue, &present_info);
	}

	vkDeviceWaitIdle(vulkan_ctx.device);
}