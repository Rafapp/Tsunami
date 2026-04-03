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

// =======================
// === Context structs ===
// =======================

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
	uint32_t            scratch_alignment;
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

// Holds one BLAS and the buffers that back it.
struct BLAS {
	VkAccelerationStructureKHR handle         = VK_NULL_HANDLE;
	VkBuffer                   buffer         = VK_NULL_HANDLE;
	VmaAllocation              buffer_alloc   = VK_NULL_HANDLE;
	VkDeviceAddress            device_address = 0;
};

struct AccelerationStructureContext {
	std::vector<BLAS> blases;

	VkAccelerationStructureKHR tlas              = VK_NULL_HANDLE;
	VkBuffer                   tlas_buffer       = VK_NULL_HANDLE;
	VmaAllocation              tlas_buffer_alloc = VK_NULL_HANDLE;
	VkDeviceAddress            tlas_address      = 0;
} as_ctx;

// ============================================================
// === Buffer helpers
// ============================================================

// Create a GPU-side buffer (device-local, not host-visible).
// Usage flags control what the buffer may be used for.
static VkBuffer create_gpu_buffer(VmaAllocator allocator, VkDeviceSize size,
                                  VkBufferUsageFlags usage, VmaAllocation& out_alloc,
                                  VkDeviceSize alignment = 0) {
	VkBufferCreateInfo buf_info{};
	buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buf_info.size  = size;
	buf_info.usage = usage;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkBuffer buffer;
	VkResult result =
	    alignment > 0 ?
	        vmaCreateBufferWithAlignment(allocator, &buf_info, &alloc_info, alignment, &buffer,
	                                     &out_alloc, nullptr) :
	        vmaCreateBuffer(allocator, &buf_info, &alloc_info, &buffer, &out_alloc, nullptr);

	if (result != VK_SUCCESS)
		throw std::runtime_error("failed to create gpu buffer");
	return buffer;
}

// Create a CPU→GPU buffer pre-filled with data.
static VkBuffer create_and_upload_buffer(VmaAllocator allocator, VkDeviceSize size,
                                         const void* data, VkBufferUsageFlags usage,
                                         VmaAllocation& out_alloc) {
	VkBufferCreateInfo buf_info{};
	buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buf_info.size  = size;
	buf_info.usage = usage;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
	alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

	VkBuffer          buffer;
	VmaAllocationInfo info;
	if (vmaCreateBuffer(allocator, &buf_info, &alloc_info, &buffer, &out_alloc, &info) !=
	    VK_SUCCESS)
		throw std::runtime_error("failed to create and upload buffer");
	memcpy(info.pMappedData, data, size);
	return buffer;
}

// Return the device address of any buffer that was created with
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
static VkDeviceAddress get_buffer_device_address(VkDevice device, VkBuffer buffer) {
	VkBufferDeviceAddressInfo info{};
	info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	info.buffer = buffer;
	return vkGetBufferDeviceAddress(device, &info);
}

// ============================================================
// === Image helpers
// ============================================================

static VkImageView create_image_view_2d(VkDevice device, VkImage image, VkFormat format) {
	VkImageViewCreateInfo view_info{};
	view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image                           = image;
	view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format                          = format;
	view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	view_info.subresourceRange.baseMipLevel   = 0;
	view_info.subresourceRange.levelCount     = 1;
	view_info.subresourceRange.baseArrayLayer = 0;
	view_info.subresourceRange.layerCount     = 1;

	VkImageView view;
	if (vkCreateImageView(device, &view_info, nullptr, &view) != VK_SUCCESS)
		throw std::runtime_error("failed to create image view");
	return view;
}

// Transition an image layout with a full pipeline barrier.
static void transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout,
                                    VkImageLayout new_layout, VkAccessFlags src_access,
                                    VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                                    VkPipelineStageFlags dst_stage) {
	VkImageMemoryBarrier barrier{};
	barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout           = old_layout;
	barrier.newLayout           = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	barrier.srcAccessMask       = src_access;
	barrier.dstAccessMask       = dst_access;

	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// ============================================================
// === Command helpers
// ============================================================

// Allocate a one-shot command buffer, begin it, and return it.
static VkCommandBuffer begin_one_time_cmd(VkDevice device, VkCommandPool pool) {
	VkCommandBufferAllocateInfo alloc_info{};
	alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	alloc_info.commandPool        = pool;
	alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	alloc_info.commandBufferCount = 1;

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(device, &alloc_info, &cmd);

	VkCommandBufferBeginInfo begin_info{};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &begin_info);

	return cmd;
}

// End, submit, wait idle, and free a one-shot command buffer.
static void end_one_time_cmd(VkDevice device, VkCommandPool pool, VkQueue queue,
                             VkCommandBuffer cmd) {
	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit{};
	submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &cmd;
	vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);
	vkFreeCommandBuffers(device, pool, 1, &cmd);
}

// ============================================================
// === Shader compilation
// ============================================================

static std::vector<uint32_t> compile_slang_shader(const std::string& path,
                                                  const std::string& entry_point) {
	SlangSession*        session = spCreateSession(nullptr);
	SlangCompileRequest* request = spCreateCompileRequest(session);

	int target_idx = spAddCodeGenTarget(request, SLANG_SPIRV);
	spSetTargetProfile(request, target_idx, spFindProfile(session, "spirv_1_3"));

	int unit_idx = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
	spAddTranslationUnitSourceFile(request, unit_idx, path.c_str());
	spAddEntryPoint(request, unit_idx, entry_point.c_str(), SLANG_STAGE_COMPUTE);

	SlangResult result      = spCompile(request);
	const char* diagnostics = spGetDiagnosticOutput(request);
	if (diagnostics && diagnostics[0] != '\0')
		std::cerr << "[SLANG] " << path << ":\n" << diagnostics << "\n";

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

// ============================================================
// === Descriptor layout helpers
// ============================================================

static VkDescriptorSetLayoutBinding make_image_binding(uint32_t binding) {
	VkDescriptorSetLayoutBinding b{};
	b.binding         = binding;
	b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	b.descriptorCount = 1;
	b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	return b;
}

static VkDescriptorSetLayoutBinding make_buffer_binding(uint32_t binding) {
	VkDescriptorSetLayoutBinding b{};
	b.binding         = binding;
	b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	b.descriptorCount = 1;
	b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	return b;
}

static VkDescriptorSetLayoutBinding make_as_binding(uint32_t binding) {
	VkDescriptorSetLayoutBinding b{};
	b.binding         = binding;
	b.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	b.descriptorCount = 1;
	b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	return b;
}

// ============================================================
// === Acceleration structure helpers
// ============================================================

// Build flags used consistently for all AS builds.
static constexpr VkBuildAccelerationStructureFlagsKHR AS_BUILD_FLAGS =
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

// All AS-related buffers need these two usage flags in addition to their own.
static constexpr VkBufferUsageFlags AS_BUFFER_USAGE =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

static constexpr VkBufferUsageFlags AS_INPUT_BUFFER_USAGE =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

static constexpr VkBufferUsageFlags SCRATCH_BUFFER_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

// Build a BLAS from a GPUMesh's vertex + index data.
// Vertices are assumed to be tightly packed float3 positions.
static BLAS build_blas(VmaAllocator allocator, VkDevice device, VkCommandPool pool, VkQueue queue,
                       const void* vertices, uint32_t vertex_count, uint32_t vertex_stride,
                       const void* indices, uint32_t index_count) {
	std::cout << "[build_blas] enter:"
	          << " vertex_count=" << vertex_count << " vertex_stride=" << vertex_stride
	          << " index_count=" << index_count << "\n";

	const VkDeviceSize vertex_size = (VkDeviceSize) vertex_stride * vertex_count;
	const VkDeviceSize index_size  = sizeof(uint32_t) * index_count;

	std::cout << "[build_blas] buffer sizes: vertex=" << vertex_size << " index=" << index_size
	          << "\n";

	// --- Upload vertex and index data ---
	VmaAllocation vertex_alloc, index_alloc;

	std::cout << "[build_blas] creating vertex buffer...\n";
	VkBuffer vertex_buf = create_and_upload_buffer(
	    allocator, vertex_size, vertices, AS_INPUT_BUFFER_USAGE | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	    vertex_alloc);
	std::cout << "[build_blas] vertex buffer OK\n";

	std::cout << "[build_blas] creating index buffer...\n";
	VkBuffer index_buf = create_and_upload_buffer(
	    allocator, index_size, indices, AS_INPUT_BUFFER_USAGE | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	    index_alloc);
	std::cout << "[build_blas] index buffer OK\n";

	VkDeviceAddress vertex_addr = get_buffer_device_address(device, vertex_buf);
	VkDeviceAddress index_addr  = get_buffer_device_address(device, index_buf);

	if (vertex_addr == 0)
		throw std::runtime_error("[build_blas] vertex buffer address is null");
	if (index_addr == 0)
		throw std::runtime_error("[build_blas] index buffer address is null");

	// --- Describe geometry ---
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
	triangles.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	triangles.vertexData   = {.deviceAddress = vertex_addr};
	triangles.vertexStride = vertex_stride;
	triangles.maxVertex    = vertex_count - 1;
	triangles.indexType    = VK_INDEX_TYPE_UINT32;
	triangles.indexData    = {.deviceAddress = index_addr};

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.geometry.triangles = triangles;
	geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;

	// --- Query build sizes ---
	const uint32_t triangle_count = index_count / 3;

	VkAccelerationStructureBuildGeometryInfoKHR build_info{};
	build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	build_info.flags         = AS_BUILD_FLAGS;
	build_info.geometryCount = 1;
	build_info.pGeometries   = &geometry;

	std::cout << "[build_blas] querying build sizes (triangle_count=" << triangle_count << ")...\n";

	VkAccelerationStructureBuildSizesInfoKHR size_info{};
	size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &build_info, &triangle_count, &size_info);

	if (size_info.accelerationStructureSize == 0)
		throw std::runtime_error("[build_blas] AS size is zero — geometry is malformed");
	if (size_info.buildScratchSize == 0)
		throw std::runtime_error("[build_blas] scratch size is zero");

	// --- Allocate BLAS storage and scratch buffers ---
	BLAS blas;

	std::cout << "[build_blas] creating AS storage buffer...\n";
	blas.buffer = create_gpu_buffer(allocator, size_info.accelerationStructureSize, AS_BUFFER_USAGE,
	                                blas.buffer_alloc);
	std::cout << "[build_blas] AS storage buffer OK\n";

	VmaAllocation scratch_alloc;
	std::cout << "[build_blas] creating scratch buffer...\n";
	VkBuffer scratch_buf =
	    create_gpu_buffer(allocator, size_info.buildScratchSize, SCRATCH_BUFFER_USAGE,
	                      scratch_alloc, vulkan_ctx.scratch_alignment);
	std::cout << "[build_blas] scratch buffer OK\n";

	// --- Create the BLAS object ---
	std::cout << "[build_blas] creating AS object...\n";
	VkAccelerationStructureCreateInfoKHR as_create_info{};
	as_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	as_create_info.buffer = blas.buffer;
	as_create_info.size   = size_info.accelerationStructureSize;
	as_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

	VkResult create_result =
	    vkCreateAccelerationStructureKHR(device, &as_create_info, nullptr, &blas.handle);
	std::cout << "[build_blas] vkCreateAccelerationStructureKHR result=" << create_result << "\n";
	if (create_result != VK_SUCCESS)
		throw std::runtime_error("[build_blas] failed to create BLAS, VkResult=" +
		                         std::to_string(create_result));

	VkDeviceAddress scratch_addr = get_buffer_device_address(device, scratch_buf);
	if (scratch_addr == 0)
		throw std::runtime_error("[build_blas] scratch buffer address is null");

	// --- Record and submit build ---
	build_info.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build_info.dstAccelerationStructure  = blas.handle;
	build_info.scratchData.deviceAddress = scratch_addr;

	VkAccelerationStructureBuildRangeInfoKHR range_info{};
	range_info.primitiveCount                               = triangle_count;
	range_info.primitiveOffset                              = 0;
	range_info.firstVertex                                  = 0;
	range_info.transformOffset                              = 0;
	const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range_info;

	std::cout << "[build_blas] beginning command buffer...\n";
	VkCommandBuffer cmd = begin_one_time_cmd(device, pool);
	std::cout << "[build_blas] recording vkCmdBuildAccelerationStructuresKHR...\n";
	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &p_range);
	std::cout << "[build_blas] submitting...\n";
	end_one_time_cmd(device, pool, queue, cmd);
	std::cout << "[build_blas] build complete\n";

	// --- Get device address ---
	VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
	addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	addr_info.accelerationStructure = blas.handle;
	blas.device_address = vkGetAccelerationStructureDeviceAddressKHR(device, &addr_info);

	vmaDestroyBuffer(allocator, scratch_buf, scratch_alloc);
	vmaDestroyBuffer(allocator, vertex_buf, vertex_alloc);
	vmaDestroyBuffer(allocator, index_buf, index_alloc);

	std::cout << "[build_blas] done\n";
	return blas;
}

// Build a TLAS from a list of already-built BLASes.
// Each BLAS gets an identity transform; pass actual transforms if you need them.
static void build_tlas(VmaAllocator allocator, VkDevice device, VkCommandPool pool, VkQueue queue,
                       const std::vector<BLAS>& blases) {
	// --- One instance per BLAS ---
	std::vector<VkAccelerationStructureInstanceKHR> instances;
	instances.reserve(blases.size());

	for (uint32_t i = 0; i < (uint32_t) blases.size(); ++i) {
		VkAccelerationStructureInstanceKHR inst{};
		// Identity transform (row-major 3x4)
		inst.transform                              = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}};
		inst.instanceCustomIndex                    = i;
		inst.mask                                   = 0xFF;
		inst.instanceShaderBindingTableRecordOffset = 0;
		inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		inst.accelerationStructureReference = blases[i].device_address;
		instances.push_back(inst);
	}

	const VkDeviceSize instance_size =
	    sizeof(VkAccelerationStructureInstanceKHR) * instances.size();

	VmaAllocation instance_alloc;
	VkBuffer instance_buf = create_and_upload_buffer(allocator, instance_size, instances.data(),
	                                                 AS_INPUT_BUFFER_USAGE, instance_alloc);

	VkDeviceAddress instance_addr = get_buffer_device_address(device, instance_buf);

	// --- Describe TLAS geometry ---
	VkAccelerationStructureGeometryInstancesDataKHR instances_data{};
	instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	instances_data.data.deviceAddress = instance_addr;

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.geometry.instances = instances_data;

	// --- Query build sizes ---
	VkAccelerationStructureBuildGeometryInfoKHR build_info{};
	build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build_info.flags         = AS_BUILD_FLAGS;
	build_info.geometryCount = 1;
	build_info.pGeometries   = &geometry;

	const uint32_t                           instance_count = (uint32_t) instances.size();
	VkAccelerationStructureBuildSizesInfoKHR size_info{};
	size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &build_info, &instance_count, &size_info);

	// --- Allocate TLAS ---
	VmaAllocation scratch_alloc;
	as_ctx.tlas_buffer = create_gpu_buffer(allocator, size_info.accelerationStructureSize,
	                                       AS_BUFFER_USAGE, as_ctx.tlas_buffer_alloc);
	VkBuffer scratch_buf =
	    create_gpu_buffer(allocator, size_info.buildScratchSize, SCRATCH_BUFFER_USAGE,
	                      scratch_alloc, vulkan_ctx.scratch_alignment);

	VkAccelerationStructureCreateInfoKHR as_create_info{};
	as_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	as_create_info.buffer = as_ctx.tlas_buffer;
	as_create_info.size   = size_info.accelerationStructureSize;
	as_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

	if (vkCreateAccelerationStructureKHR(device, &as_create_info, nullptr, &as_ctx.tlas) !=
	    VK_SUCCESS)
		throw std::runtime_error("failed to create TLAS");

	// --- Build ---
	build_info.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build_info.dstAccelerationStructure  = as_ctx.tlas;
	build_info.scratchData.deviceAddress = get_buffer_device_address(device, scratch_buf);

	VkAccelerationStructureBuildRangeInfoKHR range_info{};
	range_info.primitiveCount = instance_count;

	const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range_info;

	VkCommandBuffer cmd = begin_one_time_cmd(device, pool);
	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &p_range);
	end_one_time_cmd(device, pool, queue, cmd);

	VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
	addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	addr_info.accelerationStructure = as_ctx.tlas;
	as_ctx.tlas_address = vkGetAccelerationStructureDeviceAddressKHR(device, &addr_info);

	vmaDestroyBuffer(allocator, scratch_buf, scratch_alloc);
	vmaDestroyBuffer(allocator, instance_buf, instance_alloc);
}

// =======================
// === App constructor ===
// =======================

App::App() {
	// ======================
	// === 0. Scene setup ===
	// ======================

	m_scene = std::make_unique<Scene>();

	m_scene->m_camera = Camera(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f),
	                           glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 0.1f, 100.0f);

	// Floor (white)
	m_scene->m_shapes.push_back(std::make_unique<Quad>(
	    Transform(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f), glm::vec3(2.0f, 1.0f, 2.0f)),
	    std::make_shared<Lambert>(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Ceiling (white)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(180.0f, 0.0f, 0.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           std::make_shared<Lambert>(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Back wall (white)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(90.0f, 0.0f, 0.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           std::make_shared<Lambert>(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Left wall (red)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 90.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           std::make_shared<Lambert>(glm::vec3(0.8f, 0.1f, 0.1f))));

	// Right wall (green)
	m_scene->m_shapes.push_back(
	    std::make_unique<Quad>(Transform(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -90.0f),
	                                     glm::vec3(2.0f, 1.0f, 2.0f)),
	                           std::make_shared<Lambert>(glm::vec3(0.1f, 0.8f, 0.1f))));

	// Area light (emissive quad on ceiling)
	m_scene->m_shapes.push_back(std::make_unique<Quad>(
	    Transform(glm::vec3(0.0f, 0.99f, 0.0f), glm::vec3(180.0f, 0.0f, 0.0f),
	              glm::vec3(2.5f, 1.0f, 2.5f)),
	    std::make_shared<Lambert>(glm::vec3(1.0f), glm::vec3(15.0f), 0.25f)));

	// Tall box
	m_scene->m_shapes.push_back(
	    std::make_unique<Box>(Transform(glm::vec3(-0.35f, -0.35f, -0.4f),
	                                    glm::vec3(0.0f, 15.0f, 0.0f), glm::vec3(0.3f, 0.65f, 0.3f)),
	                          std::make_shared<Lambert>(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Short box
	m_scene->m_shapes.push_back(std::make_unique<Box>(
	    Transform(glm::vec3(0.35f, -0.65f, -0.2f), glm::vec3(0.0f, -15.0f, 0.0f),
	              glm::vec3(0.3f, 0.35f, 0.3f)),
	    std::make_shared<Lambert>(glm::vec3(0.8f, 0.8f, 0.8f))));

	// Teapot (mesh)
	m_scene->m_meshes.push_back(std::make_unique<Mesh>(
	    "resources/meshes/teapot.obj",
	    Transform(glm::vec3(0.0f, -0.5f, 0.5f), glm::vec3(0.0f, 45.0f, 0.0f), glm::vec3(0.5f)),
	    std::make_shared<Lambert>(glm::vec3(0.8f, 0.8f, 0.8f))));

	// --- Pack scene data for GPU ---
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

	// ========================================
	// === I. Load vulkan function pointers ===
	// ========================================

	if (volkInitialize() != VK_SUCCESS)
		throw std::runtime_error("failed to initialize volk");
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

	{
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
	}

	if (glfwCreateWindowSurface(vulkan_ctx.instance, m_window->handle(), nullptr,
	                            &vulkan_ctx.surface) != VK_SUCCESS)
		throw std::runtime_error("failed to create window surface");
	std::cout << "[INFO] Created window surface\n";

	{
		std::vector<const char*> device_extensions = {
		    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		    VK_KHR_RAY_QUERY_EXTENSION_NAME,
		    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
		};

		auto phys_ret = vkb::PhysicalDeviceSelector(vulkan_ctx.instance)
		                    .set_surface(vulkan_ctx.surface)
		                    .set_minimum_version(1, 3)
		                    .add_required_extensions(device_extensions)
		                    .select();

		if (!phys_ret) {
			// Print per-device capability info to help diagnose the failure
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
		vulkan_ctx.phys_device = phys_ret.value();
	}

	{
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(vulkan_ctx.phys_device.physical_device, &props);
		std::cout << "[INFO] Selected physical device: " << props.deviceName << "\n";
	}

	{
		VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
		as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		as_features.accelerationStructure = VK_TRUE;

		VkPhysicalDeviceRayQueryFeaturesKHR rq_features{};
		rq_features.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
		rq_features.rayQuery = VK_TRUE;

		VkPhysicalDeviceBufferDeviceAddressFeatures bda_features{};
		bda_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		bda_features.bufferDeviceAddress = VK_TRUE;

		auto dev_ret = vkb::DeviceBuilder{vulkan_ctx.phys_device}
		                   .add_pNext(&as_features)
		                   .add_pNext(&rq_features)
		                   .add_pNext(&bda_features)
		                   .build();
		if (!dev_ret)
			throw std::runtime_error("failed to create logical device");
		vulkan_ctx.log_device = dev_ret.value();
		vulkan_ctx.device     = vulkan_ctx.log_device.device;
		volkLoadDevice(vulkan_ctx.device);
		std::cout << "[INFO] Created logical device\n";
	}

	{
		VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props{};
		as_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
		VkPhysicalDeviceProperties2 props2{};
		props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		props2.pNext = &as_props;
		vkGetPhysicalDeviceProperties2(vulkan_ctx.phys_device.physical_device, &props2);

		vulkan_ctx.scratch_alignment = as_props.minAccelerationStructureScratchOffsetAlignment;
	}

	{
		auto queue_ret = vulkan_ctx.log_device.get_queue(vkb::QueueType::graphics);
		if (!queue_ret)
			throw std::runtime_error("failed to get graphics queue");
		vulkan_ctx.graphics_queue = queue_ret.value();

		auto family_ret = vulkan_ctx.log_device.get_queue_index(vkb::QueueType::graphics);
		if (!family_ret)
			throw std::runtime_error("failed to get graphics queue family");
		vulkan_ctx.graphics_queue_family = family_ret.value();
		std::cout << "[INFO] Acquired graphics queue (family " << vulkan_ctx.graphics_queue_family
		          << ")\n";
	}

	// ==========================
	// === IV. Initialize VMA ===
	// ==========================

	{
		VmaVulkanFunctions vma_funcs{};
		vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
		vma_funcs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

		VmaAllocatorCreateInfo vma_info{};
		vma_info.instance         = vulkan_ctx.instance.instance;
		vma_info.physicalDevice   = vulkan_ctx.phys_device.physical_device;
		vma_info.device           = vulkan_ctx.device;
		vma_info.vulkanApiVersion = VK_API_VERSION_1_3;
		// Required for get_buffer_device_address and AS builds
		vma_info.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
		vma_info.pVulkanFunctions = &vma_funcs;

		if (vmaCreateAllocator(&vma_info, &render_target_ctx.allocator) != VK_SUCCESS)
			throw std::runtime_error("failed to create VMA allocator");
		std::cout << "[INFO] Created VMA allocator\n";
	}

	VmaAllocator allocator = render_target_ctx.allocator;

	// ===================================
	// === V. Create swapchain context ===
	// ===================================

	{
		auto swap_ret =
		    vkb::SwapchainBuilder{vulkan_ctx.log_device}
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

		auto views_ret = swapchain_ctx.swapchain.get_image_views();
		if (!views_ret)
			throw std::runtime_error("failed to get swapchain image views");
		swapchain_ctx.image_views = views_ret.value();
		std::cout << "[INFO] Acquired " << swapchain_ctx.images.size()
		          << " swapchain images and views\n";
	}

	// ========================================
	// === VI. Create Render Target Context ===
	// ========================================

	{
		const VkExtent3D extent = {(uint32_t) m_window->width(), (uint32_t) m_window->height(), 1};

		auto make_storage_image = [&](VkImage& out_image, VmaAllocation& out_alloc,
		                              VkImageUsageFlags extra_usage) {
			VkImageCreateInfo image_info{};
			image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			image_info.imageType     = VK_IMAGE_TYPE_2D;
			image_info.format        = VK_FORMAT_R8G8B8A8_UNORM;
			image_info.extent        = extent;
			image_info.mipLevels     = 1;
			image_info.arrayLayers   = 1;
			image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
			image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
			image_info.usage         = VK_IMAGE_USAGE_STORAGE_BIT | extra_usage;
			image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

			VmaAllocationCreateInfo alloc_info{};
			alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

			if (vmaCreateImage(allocator, &image_info, &alloc_info, &out_image, &out_alloc,
			                   nullptr) != VK_SUCCESS)
				throw std::runtime_error("failed to create storage image");
		};

		make_storage_image(render_target_ctx.storage_image, render_target_ctx.storage_image_alloc,
		                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		make_storage_image(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc, 0);

		render_target_ctx.storage_image_view = create_image_view_2d(
		    vulkan_ctx.device, render_target_ctx.storage_image, VK_FORMAT_R8G8B8A8_UNORM);
		render_target_ctx.accum_image_view = create_image_view_2d(
		    vulkan_ctx.device, render_target_ctx.accum_image, VK_FORMAT_R8G8B8A8_UNORM);

		std::cout << "[INFO] Created storage and accum images (" << extent.width << "x"
		          << extent.height << ")\n";
	}

	// =================================
	// === VI.5 Create Scene Buffers ===
	// =================================

	{
		// Storage buffers don't need device address; keep usage simple here.
		constexpr VkBufferUsageFlags SCENE_BUF = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		scene_ctx.camera_buffer = create_and_upload_buffer(
		    allocator, sizeof(GPUCamera), &gpu_camera, SCENE_BUF, scene_ctx.camera_alloc);
		scene_ctx.shapes_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUShape) * gpu_shapes.size(),
		                             gpu_shapes.data(), SCENE_BUF, scene_ctx.shapes_alloc);
		scene_ctx.material_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUMaterial) * gpu_materials.size(),
		                             gpu_materials.data(), SCENE_BUF, scene_ctx.material_alloc);

		std::cout << "[INFO] Uploaded scene buffers (shapes: " << scene_ctx.shape_count
		          << ", materials: " << scene_ctx.material_count << ")\n";
	}

	// =================================================
	// === VII. Create command pool (needed for AS) ===
	// =================================================

	// Create the command pool early so AS builds can use it below.
	{
		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = vulkan_ctx.graphics_queue_family;
		pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		if (vkCreateCommandPool(vulkan_ctx.device, &pool_info, nullptr,
		                        &command_ctx.command_pool) != VK_SUCCESS)
			throw std::runtime_error("failed to create command pool");
		std::cout << "[INFO] Created command pool\n";
	}

	// ======================================================
	// === VIII. Build BLAS per mesh, then build TLAS     ===
	// ======================================================

	// Shared vertex/index pools for all meshes — offsets tracked per mesh.
	std::vector<GPUVertex> all_vertices;
	std::vector<uint32_t>  all_indices;
	std::vector<GPUMesh>   gpu_meshes;

	for (auto& mesh : m_scene->m_meshes) {
		int vertex_offset = (int) all_vertices.size();
		int index_offset  = (int) all_indices.size();

		all_vertices.insert(all_vertices.end(), mesh->gpuVertices.begin(), mesh->gpuVertices.end());
		all_indices.insert(all_indices.end(), mesh->gpuIndices.begin(), mesh->gpuIndices.end());

		int mat_index = (int) gpu_materials.size();
		if (!mesh->m_material) {
			std::cerr << "[ERROR] mesh->m_material is null for mesh index "
			          << (&mesh - &m_scene->m_meshes[0]) << "\n";
			continue;        // or throw
		}
		std::cout << "[INFO] Packing mesh with material index " << mat_index << "\n";
		gpu_materials.push_back(mesh->m_material->pack());

		BLAS blas = build_blas(allocator, vulkan_ctx.device, command_ctx.command_pool,
		                       vulkan_ctx.graphics_queue, mesh->gpuVertices.data(),
		                       (uint32_t) mesh->gpuVertices.size(), sizeof(GPUVertex),
		                       mesh->gpuIndices.data(), (uint32_t) mesh->gpuIndices.size());

		// Store the BLAS device address in the packed GPUMesh so the shader
		// can look up the correct BLAS when a ray query hits this instance.
		GPUMesh gpu_mesh    = mesh->pack(mat_index, vertex_offset, index_offset);
		gpu_mesh.blasHandle = blas.device_address;

		as_ctx.blases.push_back(std::move(blas));
		gpu_meshes.push_back(gpu_mesh);

		std::cout << "[INFO] Built BLAS for mesh (" << mesh->gpuIndices.size() / 3 << " triangles, "
		          << mesh->gpuVertices.size() << " vertices)\n";
	}

	if (!as_ctx.blases.empty()) {
		build_tlas(allocator, vulkan_ctx.device, command_ctx.command_pool,
		           vulkan_ctx.graphics_queue, as_ctx.blases);
		std::cout << "[INFO] Built TLAS (" << as_ctx.blases.size() << " instances)\n";
	}

	// ==============================================
	// === IX. Descriptor layout, pool, and sets ===
	// ==============================================

	{
		// Bindings:
		//   0 = storage image (output)
		//   1 = accum image
		//   2 = camera buffer
		//   3 = shapes buffer
		//   4 = materials buffer
		//   5 = TLAS (only added when we actually have one)
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
		    make_image_binding(0),  make_image_binding(1),  make_buffer_binding(2),
		    make_buffer_binding(3), make_buffer_binding(4),
		};
		if (as_ctx.tlas != VK_NULL_HANDLE)
			bindings.push_back(make_as_binding(5));

		VkDescriptorSetLayoutCreateInfo dsl_info{};
		dsl_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dsl_info.bindingCount = (uint32_t) bindings.size();
		dsl_info.pBindings    = bindings.data();

		if (vkCreateDescriptorSetLayout(vulkan_ctx.device, &dsl_info, nullptr,
		                                &render_target_ctx.descriptor_set_layout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout");
		std::cout << "[INFO] Created descriptor set layout\n";
	}

	{
		std::vector<VkDescriptorPoolSize> pool_sizes = {
		    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
		    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
		};
		if (as_ctx.tlas != VK_NULL_HANDLE)
			pool_sizes.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1});

		VkDescriptorPoolCreateInfo pool_info{};
		pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.maxSets       = 1;
		pool_info.poolSizeCount = (uint32_t) pool_sizes.size();
		pool_info.pPoolSizes    = pool_sizes.data();

		if (vkCreateDescriptorPool(vulkan_ctx.device, &pool_info, nullptr,
		                           &render_target_ctx.descriptor_pool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool");

		VkDescriptorSetAllocateInfo alloc_info{};
		alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc_info.descriptorPool     = render_target_ctx.descriptor_pool;
		alloc_info.descriptorSetCount = 1;
		alloc_info.pSetLayouts        = &render_target_ctx.descriptor_set_layout;

		if (vkAllocateDescriptorSets(vulkan_ctx.device, &alloc_info,
		                             &render_target_ctx.descriptor_set) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate descriptor set");
		std::cout << "[INFO] Created descriptor pool and allocated set\n";
	}

	{
		// Collect all writes, then flush in a single vkUpdateDescriptorSets call.
		std::vector<VkWriteDescriptorSet> writes;

		VkDescriptorImageInfo out_image_info{};
		out_image_info.imageView   = render_target_ctx.storage_image_view;
		out_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkDescriptorImageInfo accum_image_info{};
		accum_image_info.imageView   = render_target_ctx.accum_image_view;
		accum_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkDescriptorBufferInfo camera_info{scene_ctx.camera_buffer, 0, sizeof(GPUCamera)};
		VkDescriptorBufferInfo shapes_info{scene_ctx.shapes_buffer, 0,
		                                   sizeof(GPUShape) * scene_ctx.shape_count};
		VkDescriptorBufferInfo material_info{scene_ctx.material_buffer, 0,
		                                     sizeof(GPUMaterial) * scene_ctx.material_count};

		auto add_image_write = [&](uint32_t binding, VkDescriptorImageInfo* img) {
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = binding;
			w.descriptorCount = 1;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			w.pImageInfo      = img;
			writes.push_back(w);
		};
		auto add_buffer_write = [&](uint32_t binding, VkDescriptorBufferInfo* buf) {
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = binding;
			w.descriptorCount = 1;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			w.pBufferInfo     = buf;
			writes.push_back(w);
		};

		add_image_write(0, &out_image_info);
		add_image_write(1, &accum_image_info);
		add_buffer_write(2, &camera_info);
		add_buffer_write(3, &shapes_info);
		add_buffer_write(4, &material_info);

		// TLAS write — must stay alive until vkUpdateDescriptorSets returns.
		VkWriteDescriptorSetAccelerationStructureKHR as_write_ext{};
		VkWriteDescriptorSet                         as_write{};
		if (as_ctx.tlas != VK_NULL_HANDLE) {
			as_write_ext.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
			as_write_ext.accelerationStructureCount = 1;
			as_write_ext.pAccelerationStructures    = &as_ctx.tlas;

			as_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			as_write.pNext           = &as_write_ext;
			as_write.dstSet          = render_target_ctx.descriptor_set;
			as_write.dstBinding      = 5;
			as_write.descriptorCount = 1;
			as_write.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
			writes.push_back(as_write);
		}

		vkUpdateDescriptorSets(vulkan_ctx.device, (uint32_t) writes.size(), writes.data(), 0,
		                       nullptr);
		std::cout << "[INFO] Updated descriptor sets\n";
	}

	// ============================================
	// === X. Create Compute Pipeline           ===
	// ============================================

	{
		auto spirv = compile_slang_shader("shaders/cursedPathTracingAlgo.slang", "main");
		std::cout << "[INFO] Compiled compute shader (" << (spirv.size() * sizeof(uint32_t))
		          << " bytes)\n";

		VkShaderModuleCreateInfo module_info{};
		module_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		module_info.codeSize = spirv.size() * sizeof(uint32_t);
		module_info.pCode    = spirv.data();

		VkShaderModule shader_module;
		if (vkCreateShaderModule(vulkan_ctx.device, &module_info, nullptr, &shader_module) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create shader module");

		// Push constants: frame, shape_count, material_count
		VkPushConstantRange push_range{};
		push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		push_range.offset     = 0;
		push_range.size       = sizeof(uint32_t) * 3;

		VkPipelineLayoutCreateInfo layout_info{};
		layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &render_target_ctx.descriptor_set_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push_range;

		if (vkCreatePipelineLayout(vulkan_ctx.device, &layout_info, nullptr,
		                           &compute_ctx.pipeline_layout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout");

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
		                             &compute_ctx.pipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create compute pipeline");

		vkDestroyShaderModule(vulkan_ctx.device, shader_module, nullptr);
		std::cout << "[INFO] Created compute pipeline\n";
	}

	// ================================================
	// === XI. Allocate main command buffer          ===
	// ================================================

	{
		VkCommandBufferAllocateInfo alloc_info{};
		alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.commandPool        = command_ctx.command_pool;
		alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandBufferCount = 1;

		if (vkAllocateCommandBuffers(vulkan_ctx.device, &alloc_info, &command_ctx.command_buffer) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to allocate command buffer");
		std::cout << "[INFO] Allocated main command buffer\n";
	}

	// === One-time image transitions ===
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);

		transition_image_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                        VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
		std::cout << "[INFO] Transitioned accum image to GENERAL\n";
	}

	// ==========================
	// === XII. Sync objects  ===
	// ==========================

	{
		VkSemaphoreCreateInfo sem_info{};
		sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fence_info{};
		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		sync_ctx.render_finished.resize(swapchain_ctx.images.size());

		if (vkCreateSemaphore(vulkan_ctx.device, &sem_info, nullptr, &sync_ctx.image_available) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create image_available semaphore");

		for (auto& sem : sync_ctx.render_finished)
			if (vkCreateSemaphore(vulkan_ctx.device, &sem_info, nullptr, &sem) != VK_SUCCESS)
				throw std::runtime_error("failed to create render_finished semaphore");

		if (vkCreateFence(vulkan_ctx.device, &fence_info, nullptr, &sync_ctx.in_flight) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create in_flight fence");

		std::cout << "[INFO] Created sync objects\n";
	}
}

// ============================================================
// === Main loop
// ============================================================

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

		VkCommandBuffer cmd = command_ctx.command_buffer;

		// Storage image: UNDEFINED → GENERAL (written by compute)
		transition_image_layout(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                        VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		// Bind pipeline and descriptors
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline_layout, 0,
		                        1, &render_target_ctx.descriptor_set, 0, nullptr);

		struct PushConstants {
			uint32_t frame;
			uint32_t shape_count;
			uint32_t material_count;
		};
		PushConstants pc{frame_number, scene_ctx.shape_count, scene_ctx.material_count};
		vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
		                   sizeof(PushConstants), &pc);

		uint32_t group_x = (m_window->width() + 15) / 16;
		uint32_t group_y = (m_window->height() + 15) / 16;
		vkCmdDispatch(cmd, group_x, group_y, 1);
		frame_number++;

		// Storage image: GENERAL → TRANSFER_SRC
		transition_image_layout(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_GENERAL,
		                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
		                        VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                        VK_PIPELINE_STAGE_TRANSFER_BIT);

		// Swapchain image: UNDEFINED → TRANSFER_DST
		transition_image_layout(cmd, swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_UNDEFINED,
		                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
		                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                        VK_PIPELINE_STAGE_TRANSFER_BIT);

		// Blit storage → swapchain
		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.srcOffsets[0]  = {0, 0, 0};
		blit.srcOffsets[1]  = {(int32_t) m_window->width(), (int32_t) m_window->height(), 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.dstOffsets[0]  = {0, 0, 0};
		blit.dstOffsets[1]  = {(int32_t) swapchain_ctx.swapchain.extent.width,
		                       (int32_t) swapchain_ctx.swapchain.extent.height, 1};

		vkCmdBlitImage(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
		               &blit, VK_FILTER_NEAREST);

		// Swapchain image: TRANSFER_DST → PRESENT_SRC
		transition_image_layout(
		    cmd, swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
		    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

		vkEndCommandBuffer(cmd);

		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		VkSubmitInfo submit_info{};
		submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount   = 1;
		submit_info.pWaitSemaphores      = &sync_ctx.image_available;
		submit_info.pWaitDstStageMask    = &wait_stage;
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &cmd;
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