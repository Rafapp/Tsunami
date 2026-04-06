#include <array>
#include <iostream>
#include <vector>

#define VOLK_IMPLEMENTATION
#include "volk.h"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "VkBootstrap.h"
#include "slang.h"
#include "tsunami/app/app.h"
#include "tsunami/camera/fly_camera.h"
#include "vk_mem_alloc.h"

// ============================================================
// === OpenPBR LUT data — C++ array mode
// ============================================================
// The shader uses OPENPBR_USE_TEXTURE_LUTS=1. Here on the CPU side we include
// openpbr in C++ + array mode (OPENPBR_USE_TEXTURE_LUTS=0) so the raw LUT
// data arrays are accessible as plain C++ statics for GPU upload.
#define OPENPBR_LANGUAGE_TARGET_CPP 1
#define OPENPBR_USE_TEXTURE_LUTS 0
#define OPENPBR_ENERGY_TABLES_USE_UINT16 0        // uint32 for broadest platform compat
#include "openpbr.h"
#include <glm/glm.hpp>
// Raw data headers — these define the C arrays we will upload.
// HOW TO USE:
//   1. Open openpbr_data_constants.h     → table resolution constants
//   2. Open impl/data/openpbr_energy_arrays.h → the 7 energy table uint32 arrays
//   3. Open impl/data/openpbr_ltc_array.h     → the float3 LTC table array
// Then fill in upload_openpbr_luts() below.
#include "impl/data/openpbr_energy_arrays.h"
#include "impl/data/openpbr_ltc_array.h"
#include "openpbr_data_constants.h"
// Remove macros so they don't leak.
#undef OPENPBR_LANGUAGE_TARGET_CPP
#undef OPENPBR_USE_TEXTURE_LUTS
#undef OPENPBR_ENERGY_TABLES_USE_UINT16

// ============================================================
// === LUT / texture structs
// ============================================================
struct LutTexture2D {
	VkImage       image = VK_NULL_HANDLE;
	VmaAllocation alloc = VK_NULL_HANDLE;
	VkImageView   view  = VK_NULL_HANDLE;
	uint32_t      width = 0, height = 0;
};
struct LutTexture3D {
	VkImage       image = VK_NULL_HANDLE;
	VmaAllocation alloc = VK_NULL_HANDLE;
	VkImageView   view  = VK_NULL_HANDLE;
	uint32_t      width = 0, height = 0, depth = 0;
};
static constexpr uint32_t NUM_LUTS              = 8;
static constexpr uint32_t MAX_MATERIAL_TEXTURES = 256;

// =======================
// === Context structs ===
// =======================
struct SceneContext {
	void*         camera_mapped   = nullptr;
	VkBuffer      camera_buffer   = VK_NULL_HANDLE;
	VmaAllocation camera_alloc    = VK_NULL_HANDLE;
	VkBuffer      material_buffer = VK_NULL_HANDLE;
	VmaAllocation material_alloc  = VK_NULL_HANDLE;
	uint32_t      material_count  = 0;
	VkBuffer      mesh_buffer     = VK_NULL_HANDLE;
	VmaAllocation mesh_alloc      = VK_NULL_HANDLE;
	VkBuffer      vertex_buffer   = VK_NULL_HANDLE;
	VmaAllocation vertex_alloc    = VK_NULL_HANDLE;
	VkBuffer      index_buffer    = VK_NULL_HANDLE;
	VmaAllocation index_alloc     = VK_NULL_HANDLE;
	uint32_t      mesh_count      = 0;
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
	std::vector<bool>        image_initialized;
	VkFormat                 image_format;
	VkExtent2D               extent;
} swapchain_ctx;

struct RenderTargetContext {
	VmaAllocator                       allocator;
	VkImage                            storage_image;
	VmaAllocation                      storage_image_alloc;
	VkImageView                        storage_image_view;
	bool                               storage_image_initialized = false;
	VkImage                            accum_image;
	VmaAllocation                      accum_image_alloc;
	VkImageView                        accum_image_view;
	VkImage                            dummy_image_2d;
	VmaAllocation                      dummy_image_2d_alloc;
	VkImageView                        dummy_image_2d_view;
	VkImage                            dummy_image_3d;
	VmaAllocation                      dummy_image_3d_alloc;
	VkImageView                        dummy_image_3d_view;
	VkSampler                          sampler = VK_NULL_HANDLE;
	std::array<LutTexture2D, NUM_LUTS> lut_textures_2d;
	std::array<LutTexture3D, NUM_LUTS> lut_textures_3d;
	std::vector<VkImage>               mat_images;
	std::vector<VmaAllocation>         mat_allocs;
	std::vector<VkImageView>           mat_views;
	VkDescriptorSetLayout              descriptor_set_layout;
	VkDescriptorPool                   descriptor_pool;
	VkDescriptorSet                    descriptor_set;
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
	std::vector<VkSemaphore> image_available, render_finished;
	VkFence                  in_flight;
} sync_ctx;

struct BLAS {
	VkAccelerationStructureKHR handle         = VK_NULL_HANDLE;
	VkBuffer                   buffer         = VK_NULL_HANDLE;
	VmaAllocation              buffer_alloc   = VK_NULL_HANDLE;
	VkDeviceAddress            device_address = 0;
};
struct AccelerationStructureContext {
	std::vector<BLAS>          blases;
	VkAccelerationStructureKHR tlas              = VK_NULL_HANDLE;
	VkBuffer                   tlas_buffer       = VK_NULL_HANDLE;
	VmaAllocation              tlas_buffer_alloc = VK_NULL_HANDLE;
	VkDeviceAddress            tlas_address      = 0;
} as_ctx;

// ============================================================
// === Buffer helpers
// ============================================================
static VkBuffer create_gpu_buffer(VmaAllocator alloc, VkDeviceSize sz, VkBufferUsageFlags usage,
                                  VmaAllocation& out, VkDeviceSize align = 0) {
	VkBufferCreateInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bi.size  = sz;
	bi.usage = usage;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	VkBuffer buf;
	VkResult r = align > 0 ?
	                 vmaCreateBufferWithAlignment(alloc, &bi, &ai, align, &buf, &out, nullptr) :
	                 vmaCreateBuffer(alloc, &bi, &ai, &buf, &out, nullptr);
	if (r != VK_SUCCESS)
		throw std::runtime_error("failed to create gpu buffer");
	return buf;
}
static VkBuffer create_and_upload_buffer(VmaAllocator alloc, VkDeviceSize sz, const void* data,
                                         VkBufferUsageFlags usage, VmaAllocation& out) {
	VkBufferCreateInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bi.size  = sz;
	bi.usage = usage;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
	ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBuffer          buf;
	VmaAllocationInfo info;
	if (vmaCreateBuffer(alloc, &bi, &ai, &buf, &out, &info) != VK_SUCCESS)
		throw std::runtime_error("failed to create+upload buffer");
	memcpy(info.pMappedData, data, sz);
	return buf;
}
static VkDeviceAddress get_bda(VkDevice dev, VkBuffer buf) {
	VkBufferDeviceAddressInfo i{};
	i.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	i.buffer = buf;
	return vkGetBufferDeviceAddress(dev, &i);
}

// ============================================================
// === Image helpers
// ============================================================
static VkImageView create_image_view(VkDevice dev, VkImage img, VkFormat fmt, VkImageViewType type,
                                     VkImageAspectFlags asp = VK_IMAGE_ASPECT_COLOR_BIT) {
	VkImageViewCreateInfo vi{};
	vi.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vi.image            = img;
	vi.viewType         = type;
	vi.format           = fmt;
	vi.subresourceRange = {asp, 0, 1, 0, 1};
	VkImageView v       = VK_NULL_HANDLE;
	if (vkCreateImageView(dev, &vi, nullptr, &v) != VK_SUCCESS)
		throw std::runtime_error("failed to create image view");
	return v;
}
static void transition_layout(VkCommandBuffer cmd, VkImage img, VkImageLayout old, VkImageLayout n,
                              VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags sp,
                              VkPipelineStageFlags dp) {
	VkImageMemoryBarrier b{};
	b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	b.oldLayout           = old;
	b.newLayout           = n;
	b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image                                       = img;
	b.subresourceRange                            = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	b.srcAccessMask                               = src;
	b.dstAccessMask                               = dst;
	vkCmdPipelineBarrier(cmd, sp, dp, 0, 0, nullptr, 0, nullptr, 1, &b);
}
// Keep old name as alias for callers
static void transition_image_layout(VkCommandBuffer cmd, VkImage img, VkImageLayout o,
                                    VkImageLayout n, VkAccessFlags sa, VkAccessFlags da,
                                    VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
	transition_layout(cmd, img, o, n, sa, da, ss, ds);
}

// ============================================================
// === Command helpers
// ============================================================
static VkCommandBuffer begin_one_time_cmd(VkDevice dev, VkCommandPool pool) {
	VkCommandBufferAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = pool;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(dev, &ai, &cmd);
	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &bi);
	return cmd;
}
static void end_one_time_cmd(VkDevice dev, VkCommandPool pool, VkQueue q, VkCommandBuffer cmd) {
	vkEndCommandBuffer(cmd);
	VkSubmitInfo si{};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cmd;
	vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(q);
	vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

// ============================================================
// === Generic RGBA32F image upload
// ============================================================
static void upload_rgba32f_2d(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const float* rgba, uint32_t w, uint32_t h, LutTexture2D& out) {
	VkDeviceSize  sz = (VkDeviceSize) w * h * 4 * sizeof(float);
	VmaAllocation sa;
	VkBuffer stg = create_and_upload_buffer(alloc, sz, rgba, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);
	VkImageCreateInfo ii{};
	ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ii.imageType     = VK_IMAGE_TYPE_2D;
	ii.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent        = {w, h, 1};
	ii.mipLevels     = 1;
	ii.arrayLayers   = 1;
	ii.samples       = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
	ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(alloc, &ii, &ai, &out.image, &out.alloc, nullptr);
	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkBufferImageCopy r{};
	r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	r.imageExtent      = {w, h, 1};
	vkCmdCopyBufferToImage(cmd, stg, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	end_one_time_cmd(dev, pool, q, cmd);
	out.view =
	    create_image_view(dev, out.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D);
	out.width  = w;
	out.height = h;
	vmaDestroyBuffer(alloc, stg, sa);
}
static void upload_rgba32f_3d(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const float* rgba, uint32_t w, uint32_t h, uint32_t d,
                              LutTexture3D& out) {
	VkDeviceSize  sz = (VkDeviceSize) w * h * d * 4 * sizeof(float);
	VmaAllocation sa;
	VkBuffer stg = create_and_upload_buffer(alloc, sz, rgba, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);
	VkImageCreateInfo ii{};
	ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ii.imageType     = VK_IMAGE_TYPE_3D;
	ii.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent        = {w, h, d};
	ii.mipLevels     = 1;
	ii.arrayLayers   = 1;
	ii.samples       = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
	ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(alloc, &ii, &ai, &out.image, &out.alloc, nullptr);
	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkBufferImageCopy r{};
	r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	r.imageExtent      = {w, h, d};
	vkCmdCopyBufferToImage(cmd, stg, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	end_one_time_cmd(dev, pool, q, cmd);
	out.view =
	    create_image_view(dev, out.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_VIEW_TYPE_3D);
	out.width  = w;
	out.height = h;
	out.depth  = d;
	vmaDestroyBuffer(alloc, stg, sa);
}

// =======================
// === Window resizing ===
// =======================
static uint32_t g_frame_number_ref = 0;
static void     handle_resize(uint32_t& frame_number, uint32_t fb_w, uint32_t fb_h) {
    vkDeviceWaitIdle(vulkan_ctx.device);

    uint32_t new_w = fb_w;
    uint32_t new_h = fb_h;
    if (new_w == 0 || new_h == 0)
        return;        // minimised – skip

    VmaAllocator allocator = render_target_ctx.allocator;

    // -- Destroy old swapchain image views --
    for (auto v : swapchain_ctx.image_views)
        vkDestroyImageView(vulkan_ctx.device, v, nullptr);
    swapchain_ctx.image_views.clear();

    // -- Destroy old render-target images --
    vkDestroyImageView(vulkan_ctx.device, render_target_ctx.storage_image_view, nullptr);
    vmaDestroyImage(allocator, render_target_ctx.storage_image,
	                    render_target_ctx.storage_image_alloc);
    vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_image_view, nullptr);
    vmaDestroyImage(allocator, render_target_ctx.accum_image, render_target_ctx.accum_image_alloc);

    // -- Rebuild swapchain --
    vkb::Swapchain old_swapchain = swapchain_ctx.swapchain;
    auto           swap_ret =
        vkb::SwapchainBuilder{vulkan_ctx.log_device}
            .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(new_w, new_h)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .set_old_swapchain(old_swapchain.swapchain)
            .build();
    vkb::destroy_swapchain(old_swapchain);

    if (!swap_ret) {
        std::cerr << "[RESIZE] Failed to rebuild swapchain\n";
        return;
    }
    swapchain_ctx.swapchain    = swap_ret.value();
    swapchain_ctx.image_format = swapchain_ctx.swapchain.image_format;
    auto images_ret            = swapchain_ctx.swapchain.get_images();
    if (!images_ret)
        return;
    swapchain_ctx.images = images_ret.value();
    auto views_ret       = swapchain_ctx.swapchain.get_image_views();
    if (!views_ret)
        return;
    swapchain_ctx.image_views = views_ret.value();
    swapchain_ctx.image_initialized.assign(swapchain_ctx.images.size(), false);

    // -- Recreate render-target images at new size --
    const VkExtent3D ext = {new_w, new_h, 1};

    auto make_storage = [&](VkImage& img, VmaAllocation& alloc, VkImageUsageFlags extra) {
        VkImageCreateInfo ii{};
        ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType     = VK_IMAGE_TYPE_2D;
        ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent        = ext;
        ii.mipLevels     = 1;
        ii.arrayLayers   = 1;
        ii.samples       = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | extra;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateImage(allocator, &ii, &ai, &img, &alloc, nullptr);
    };

    make_storage(render_target_ctx.storage_image, render_target_ctx.storage_image_alloc,
	                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc, 0);

    render_target_ctx.storage_image_view =
        create_image_view(vulkan_ctx.device, render_target_ctx.storage_image,
	                          VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
    render_target_ctx.accum_image_view =
        create_image_view(vulkan_ctx.device, render_target_ctx.accum_image,
	                          VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);

    // Transition accum → GENERAL
    {
        VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
        transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                      VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                     cmd);
    }

    // Update the two storage-image descriptors (bindings 0 and 1)
    {
        VkDescriptorImageInfo out_img{};
        out_img.imageView   = render_target_ctx.storage_image_view;
        out_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo accum_img{};
        accum_img.imageView   = render_target_ctx.accum_image_view;
        accum_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		                 nullptr,
		                 render_target_ctx.descriptor_set,
		                 0,
		                 0,
		                 1,
		                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		                 &out_img};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		                 nullptr,
		                 render_target_ctx.descriptor_set,
		                 1,
		                 0,
		                 1,
		                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		                 &accum_img};
        vkUpdateDescriptorSets(vulkan_ctx.device, 2, writes, 0, nullptr);
    }

    // Reset sync semaphores if the swapchain image count changed
    {
        const uint32_t n = (uint32_t) swapchain_ctx.images.size();
        if (n != (uint32_t) sync_ctx.image_available.size()) {
            for (auto s : sync_ctx.image_available)
                vkDestroySemaphore(vulkan_ctx.device, s, nullptr);
            for (auto s : sync_ctx.render_finished)
                vkDestroySemaphore(vulkan_ctx.device, s, nullptr);
            sync_ctx.image_available.resize(n);
            sync_ctx.render_finished.resize(n);
            VkSemaphoreCreateInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            for (uint32_t i = 0; i < n; ++i) {
                vkCreateSemaphore(vulkan_ctx.device, &si, nullptr, &sync_ctx.image_available[i]);
                vkCreateSemaphore(vulkan_ctx.device, &si, nullptr, &sync_ctx.render_finished[i]);
            }
        }
    }

    render_target_ctx.storage_image_initialized = false;
    frame_number                                = 0;        // reset temporal accumulation

    std::cout << "[RESIZE] " << new_w << "x" << new_h << "\n";
}

// ============================================================
// === LUT data conversion helpers
// ============================================================
// Energy table: uint32 [0..65535] → RGBA32F  (R = val/65535, GBA = 0)
static std::vector<float> uint_lut_to_rgba32f(const uint32_t* data, size_t count) {
	std::vector<float> out(count * 4, 0.f);
	for (size_t i = 0; i < count; ++i)
		out[i * 4] = float(data[i]) / 65535.f;
	return out;
}
// LTC table: float3 per texel → RGBA32F  (A = 0)
static std::vector<float> float3_lut_to_rgba32f(const float* data, size_t count_vec3) {
	std::vector<float> out(count_vec3 * 4, 0.f);
	for (size_t i = 0; i < count_vec3; ++i) {
		out[i * 4 + 0] = data[i * 3 + 0];
		out[i * 4 + 1] = data[i * 3 + 1];
		out[i * 4 + 2] = data[i * 3 + 2];
	}
	return out;
}

static void upload_openpbr_luts(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q) {
	// Requires:
	//   #include "vendors/openpbr-bsdf/openpbr_data_constants.h"
	//   #include "vendors/openpbr-bsdf/impl/data/openpbr_energy_arrays.h"
	//   #include "vendors/openpbr-bsdf/impl/data/openpbr_ltc_array.h"

	// ---------------------------------------------------------------------
	// 3D energy LUTs (32 x 32 x 32)
	// ---------------------------------------------------------------------
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize *
		                                    OpenPBR_EnergyTableSize);

		upload_rgba32f_3d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_3d[OpenPBR_LutId_IdealDielectricEnergyComplement]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_OpaqueDielectricEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize *
		                                    OpenPBR_EnergyTableSize);

		upload_rgba32f_3d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_3d[OpenPBR_LutId_OpaqueDielectricEnergyComplement]);
	}

	// ---------------------------------------------------------------------
	// 2D energy LUTs (32 x 32)
	// ---------------------------------------------------------------------
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx
		        .lut_textures_2d[OpenPBR_LutId_IdealDielectricAverageEnergyComplement]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricReflectionRatio_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealDielectricReflectionRatio]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_OpaqueDielectricAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx
		        .lut_textures_2d[OpenPBR_LutId_OpaqueDielectricAverageEnergyComplement]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealMetalEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealMetalEnergyComplement]);
	}

	// ---------------------------------------------------------------------
	// 1D energy LUT stored as thin 2D texture (32 x 1)
	// ---------------------------------------------------------------------
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealMetalAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, 1,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealMetalAverageEnergyComplement]);
	}

	// ---------------------------------------------------------------------
	// LTC LUT (32 x 32), float3 -> rgba32f
	// ---------------------------------------------------------------------
	{
		auto data = float3_lut_to_rgba32f(reinterpret_cast<const float*>(OpenPBR_LTC_Array),
		                                  OpenPBR_LTCTableSize * OpenPBR_LTCTableSize);

		upload_rgba32f_2d(alloc, dev, pool, q, data.data(), OpenPBR_LTCTableSize,
		                  OpenPBR_LTCTableSize,
		                  render_target_ctx.lut_textures_2d[OpenPBR_LutId_LTC]);
	}

	std::cout << "[INFO] Uploaded OpenPBR LUTs\n";
}

// ============================================================
// === Material texture upload
// ============================================================
static void upload_material_textures(VmaAllocator alloc, VkDevice dev, VkCommandPool pool,
                                     VkQueue                                      q,
                                     const std::vector<std::shared_ptr<Texture>>& textures) {
	auto& images = render_target_ctx.mat_images;
	auto& allocs = render_target_ctx.mat_allocs;
	auto& views  = render_target_ctx.mat_views;
	images.reserve(textures.size());
	allocs.reserve(textures.size());
	views.reserve(textures.size());

	for (const auto& tex : textures) {
		if (!tex || !tex->valid()) {
			images.push_back(VK_NULL_HANDLE);
			allocs.push_back(VK_NULL_HANDLE);
			views.push_back(VK_NULL_HANDLE);
			continue;
		}
		VkDeviceSize      sz = (VkDeviceSize) tex->width * tex->height * 4;
		VmaAllocation     sa;
		VkBuffer          stg = create_and_upload_buffer(alloc, sz, tex->pixels.data(),
		                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);
		const VkFormat    fmt = tex->is_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
		VkImageCreateInfo ii{};
		ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ii.imageType     = VK_IMAGE_TYPE_2D;
		ii.format        = fmt;
		ii.extent        = {(uint32_t) tex->width, (uint32_t) tex->height, 1};
		ii.mipLevels     = 1;
		ii.arrayLayers   = 1;
		ii.samples       = VK_SAMPLE_COUNT_1_BIT;
		ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VmaAllocationCreateInfo ai2{};
		ai2.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		VkImage       img;
		VmaAllocation imgalloc;
		if (vmaCreateImage(alloc, &ii, &ai2, &img, &imgalloc, nullptr) != VK_SUCCESS)
			throw std::runtime_error("failed to create material texture");
		VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
		transition_layout(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                  VK_PIPELINE_STAGE_TRANSFER_BIT);
		VkBufferImageCopy r{};
		r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		r.imageExtent      = {(uint32_t) tex->width, (uint32_t) tex->height, 1};
		vkCmdCopyBufferToImage(cmd, stg, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
		transition_layout(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(dev, pool, q, cmd);
		images.push_back(img);
		allocs.push_back(imgalloc);
		views.push_back(create_image_view(dev, img, fmt, VK_IMAGE_VIEW_TYPE_2D));
		vmaDestroyBuffer(alloc, stg, sa);
	}
	std::cout << "[INFO] Uploaded " << textures.size() << " material textures\n";
}

// ============================================================
// === Shader compilation
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

// =================================================
// === Rebuild pipeline (hot shader compilation) ===
// =================================================
static bool rebuild_pipeline() {
	// Recompile the shader from disk
	std::vector<uint32_t> spirv;
	try {
		spirv = compile_slang_shader(std::string(SHADERS_DIR) + "/pathtracer.slang", "main",
		                             {VENDORS_DIR});
	} catch (const std::exception& e) {
		std::cerr << "[SHADER RELOAD] Compile failed: " << e.what() << "\n";
		return false;
	}

	VkShaderModuleCreateInfo mci{};
	mci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	mci.codeSize = spirv.size() * sizeof(uint32_t);
	mci.pCode    = spirv.data();

	VkShaderModule new_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(vulkan_ctx.device, &mci, nullptr, &new_module) != VK_SUCCESS) {
		std::cerr << "[SHADER RELOAD] vkCreateShaderModule failed\n";
		return false;
	}

	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = new_module;
	stage.pName  = "main";

	VkComputePipelineCreateInfo pci{};
	pci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pci.stage  = stage;
	pci.layout = compute_ctx.pipeline_layout;

	VkPipeline new_pipeline = VK_NULL_HANDLE;
	if (vkCreateComputePipelines(vulkan_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr,
	                             &new_pipeline) != VK_SUCCESS) {
		vkDestroyShaderModule(vulkan_ctx.device, new_module, nullptr);
		std::cerr << "[SHADER RELOAD] vkCreateComputePipelines failed\n";
		return false;
	}

	vkDestroyShaderModule(vulkan_ctx.device, new_module, nullptr);

	// Swap in the new pipeline (device must be idle first)
	vkDeviceWaitIdle(vulkan_ctx.device);
	vkDestroyPipeline(vulkan_ctx.device, compute_ctx.pipeline, nullptr);
	compute_ctx.pipeline = new_pipeline;

	std::cout << "[SHADER RELOAD] Success (" << spirv.size() * 4 << " bytes)\n";
	return true;
}

// ============================================================
// === Descriptor layout helper
// ============================================================
static VkDescriptorSetLayoutBinding make_binding(uint32_t binding, VkDescriptorType type,
                                                 uint32_t count = 1) {
	VkDescriptorSetLayoutBinding b{};
	b.binding         = binding;
	b.descriptorType  = type;
	b.descriptorCount = count;
	b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	return b;
}

// ============================================================
// === Acceleration structure helpers
// ============================================================
static constexpr VkBuildAccelerationStructureFlagsKHR AS_BUILD_FLAGS =
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
static constexpr VkBufferUsageFlags AS_BUFFER_USAGE =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
static constexpr VkBufferUsageFlags AS_INPUT_BUFFER_USAGE =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
static constexpr VkBufferUsageFlags SCRATCH_BUFFER_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

static BLAS build_blas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                       const void* verts, uint32_t vc, uint32_t vstride, const void* idxs,
                       uint32_t ic) {
	VmaAllocation va, ia;
	VkBuffer      vb =
	    create_and_upload_buffer(alloc, (VkDeviceSize) vstride * vc, verts,
	                             AS_INPUT_BUFFER_USAGE | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, va);
	VkBuffer ib =
	    create_and_upload_buffer(alloc, (VkDeviceSize) sizeof(uint32_t) * ic, idxs,
	                             AS_INPUT_BUFFER_USAGE | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ia);
	VkAccelerationStructureGeometryTrianglesDataKHR tri{};
	tri.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	tri.vertexData   = {.deviceAddress = get_bda(dev, vb)};
	tri.vertexStride = vstride;
	tri.maxVertex    = vc - 1;
	tri.indexType    = VK_INDEX_TYPE_UINT32;
	tri.indexData    = {.deviceAddress = get_bda(dev, ib)};
	VkAccelerationStructureGeometryKHR geom{};
	geom.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geom.geometry.triangles = tri;
	geom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
	const uint32_t                              tc = ic / 3;
	VkAccelerationStructureBuildGeometryInfoKHR build{};
	build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	build.flags         = AS_BUILD_FLAGS;
	build.geometryCount = 1;
	build.pGeometries   = &geom;
	VkAccelerationStructureBuildSizesInfoKHR si{};
	si.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &build, &tc, &si);
	BLAS blas;
	blas.buffer =
	    create_gpu_buffer(alloc, si.accelerationStructureSize, AS_BUFFER_USAGE, blas.buffer_alloc);
	VmaAllocation sca;
	VkBuffer      scb = create_gpu_buffer(alloc, si.buildScratchSize, SCRATCH_BUFFER_USAGE, sca,
	                                      vulkan_ctx.scratch_alignment);
	VkAccelerationStructureCreateInfoKHR aci{};
	aci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	aci.buffer = blas.buffer;
	aci.size   = si.accelerationStructureSize;
	aci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	if (vkCreateAccelerationStructureKHR(dev, &aci, nullptr, &blas.handle) != VK_SUCCESS)
		throw std::runtime_error("failed to create BLAS");
	build.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.dstAccelerationStructure  = blas.handle;
	build.scratchData.deviceAddress = get_bda(dev, scb);
	VkAccelerationStructureBuildRangeInfoKHR ri{};
	ri.primitiveCount                                   = tc;
	const VkAccelerationStructureBuildRangeInfoKHR* pri = &ri;
	VkCommandBuffer                                 cmd = begin_one_time_cmd(dev, pool);
	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pri);
	end_one_time_cmd(dev, pool, q, cmd);
	VkAccelerationStructureDeviceAddressInfoKHR dai{};
	dai.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	dai.accelerationStructure = blas.handle;
	blas.device_address       = vkGetAccelerationStructureDeviceAddressKHR(dev, &dai);
	vmaDestroyBuffer(alloc, scb, sca);
	vmaDestroyBuffer(alloc, vb, va);
	vmaDestroyBuffer(alloc, ib, ia);
	return blas;
}

static VkTransformMatrixKHR glm_to_vk_transform(const glm::mat4& m) {
	// Vulkan expects a 3x4 row-major transform matrix.
	// GLM is column-major, so we write rows explicitly.
	VkTransformMatrixKHR out{};
	out.matrix[0][0] = m[0][0];
	out.matrix[0][1] = m[1][0];
	out.matrix[0][2] = m[2][0];
	out.matrix[0][3] = m[3][0];

	out.matrix[1][0] = m[0][1];
	out.matrix[1][1] = m[1][1];
	out.matrix[1][2] = m[2][1];
	out.matrix[1][3] = m[3][1];

	out.matrix[2][0] = m[0][2];
	out.matrix[2][1] = m[1][2];
	out.matrix[2][2] = m[2][2];
	out.matrix[2][3] = m[3][2];
	return out;
}

static void build_tlas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                       const std::vector<BLAS>&                  blases,
                       const std::vector<std::unique_ptr<Mesh>>& meshes) {
	if (blases.size() != meshes.size())
		throw std::runtime_error("build_tlas: blases.size() != meshes.size()");

	std::vector<VkAccelerationStructureInstanceKHR> insts;
	insts.reserve(blases.size());

	for (uint32_t i = 0; i < (uint32_t) blases.size(); ++i) {
		VkAccelerationStructureInstanceKHR inst{};
		inst.transform           = glm_to_vk_transform(meshes[i]->m_transform.m_transform);
		inst.instanceCustomIndex = i;
		inst.mask                = 0xFF;
		inst.instanceShaderBindingTableRecordOffset = 0;
		inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		inst.accelerationStructureReference = blases[i].device_address;
		insts.push_back(inst);
	}

	VmaAllocation ia;
	VkBuffer      ib =
	    create_and_upload_buffer(alloc, sizeof(VkAccelerationStructureInstanceKHR) * insts.size(),
	                             insts.data(), AS_INPUT_BUFFER_USAGE, ia);

	VkAccelerationStructureGeometryInstancesDataKHR id{};
	id.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	id.data.deviceAddress = get_bda(dev, ib);

	VkAccelerationStructureGeometryKHR geom{};
	geom.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geom.geometry.instances = id;

	VkAccelerationStructureBuildGeometryInfoKHR build{};
	build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build.flags         = AS_BUILD_FLAGS;
	build.geometryCount = 1;
	build.pGeometries   = &geom;

	const uint32_t ic = (uint32_t) insts.size();

	VkAccelerationStructureBuildSizesInfoKHR si{};
	si.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &build, &ic, &si);

	VmaAllocation sca;
	as_ctx.tlas_buffer = create_gpu_buffer(alloc, si.accelerationStructureSize, AS_BUFFER_USAGE,
	                                       as_ctx.tlas_buffer_alloc);
	VkBuffer scb       = create_gpu_buffer(alloc, si.buildScratchSize, SCRATCH_BUFFER_USAGE, sca,
	                                       vulkan_ctx.scratch_alignment);

	VkAccelerationStructureCreateInfoKHR aci{};
	aci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	aci.buffer = as_ctx.tlas_buffer;
	aci.size   = si.accelerationStructureSize;
	aci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

	if (vkCreateAccelerationStructureKHR(dev, &aci, nullptr, &as_ctx.tlas) != VK_SUCCESS)
		throw std::runtime_error("failed to create TLAS");

	build.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.dstAccelerationStructure  = as_ctx.tlas;
	build.scratchData.deviceAddress = get_bda(dev, scb);

	VkAccelerationStructureBuildRangeInfoKHR ri{};
	ri.primitiveCount                                   = ic;
	const VkAccelerationStructureBuildRangeInfoKHR* pri = &ri;

	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pri);
	end_one_time_cmd(dev, pool, q, cmd);

	VkAccelerationStructureDeviceAddressInfoKHR dai{};
	dai.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	dai.accelerationStructure = as_ctx.tlas;
	as_ctx.tlas_address       = vkGetAccelerationStructureDeviceAddressKHR(dev, &dai);

	vmaDestroyBuffer(alloc, scb, sca);
	vmaDestroyBuffer(alloc, ib, ia);
}

// =======================
// === App constructor ===
// =======================
App::App() {
	// ==============================
	// === 0. Scene setup
	// ==============================
	m_scene           = std::make_unique<Scene>();
	m_scene->m_camera = Camera(glm::vec3(0.f, 20.f, 0.f), glm::vec3(0.f, 0.f, 0.f),
	                           glm::vec3(0.f, 1.f, 0.f), 60.f, 0.1f, 10000.f);
	m_scene->load_gltf("resources/scenes/ABeautifulGame/glTF/ABeautifulGame.gltf");
	// m_scene->load_gltf("resources/scenes/Sponza/glTF/Sponza.gltf");

	// ========================================
	// === I. Vulkan function pointers
	// ========================================
	if (volkInitialize() != VK_SUCCESS)
		throw std::runtime_error("failed to initialize volk");
	std::cout << "[INFO] Volk initialized\n";

	// =========================
	// === II. Window
	// =========================
	m_window = std::make_unique<core::Window>(
	    core::WindowConfig{.width = 1280, .height = 720, .title = "tsunami 🌊"});
	std::cout << "[INFO] Window created\n";

	// ==================================
	// === III. Vulkan instance/device
	// ==================================
	{
		vkb::InstanceBuilder b;
		auto                 r = b.set_app_name("tsunami")
		             .request_validation_layers()
		             .use_default_debug_messenger()
		             .require_api_version(1, 3, 0)
		             .build();
		if (!r)
			throw std::runtime_error("failed to create Vulkan instance");
		vulkan_ctx.instance = r.value();
		volkLoadInstance(vulkan_ctx.instance.instance);
		std::cout << "[INFO] Vulkan instance created\n";
	}
	if (glfwCreateWindowSurface(vulkan_ctx.instance, m_window->handle(), nullptr,
	                            &vulkan_ctx.surface) != VK_SUCCESS)
		throw std::runtime_error("failed to create window surface");
	{
		std::vector<const char*> exts = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		                                 VK_KHR_RAY_QUERY_EXTENSION_NAME,
		                                 VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
		auto                     r    = vkb::PhysicalDeviceSelector(vulkan_ctx.instance)
		             .set_surface(vulkan_ctx.surface)
		             .set_minimum_version(1, 3)
		             .add_required_extensions(exts)
		             .select();
		if (!r)
			throw std::runtime_error("failed to select physical device");
		vulkan_ctx.phys_device = r.value();
		VkPhysicalDeviceProperties p{};
		vkGetPhysicalDeviceProperties(vulkan_ctx.phys_device.physical_device, &p);
		std::cout << "[INFO] GPU: " << p.deviceName << "\n";
	}
	{
		VkPhysicalDeviceAccelerationStructureFeaturesKHR af{};
		af.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		af.accelerationStructure = VK_TRUE;
		VkPhysicalDeviceRayQueryFeaturesKHR rf{};
		rf.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
		rf.rayQuery = VK_TRUE;
		VkPhysicalDeviceBufferDeviceAddressFeatures bf{};
		bf.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		bf.bufferDeviceAddress = VK_TRUE;
		auto r                 = vkb::DeviceBuilder{vulkan_ctx.phys_device}
		             .add_pNext(&af)
		             .add_pNext(&rf)
		             .add_pNext(&bf)
		             .build();
		if (!r)
			throw std::runtime_error("failed to create logical device");
		vulkan_ctx.log_device = r.value();
		vulkan_ctx.device     = vulkan_ctx.log_device.device;
		volkLoadDevice(vulkan_ctx.device);
		std::cout << "[INFO] Logical device created\n";
	}
	{
		VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{};
		asp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
		VkPhysicalDeviceProperties2 p2{};
		p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		p2.pNext = &asp;
		vkGetPhysicalDeviceProperties2(vulkan_ctx.phys_device.physical_device, &p2);
		vulkan_ctx.scratch_alignment = asp.minAccelerationStructureScratchOffsetAlignment;
	}
	{
		auto qr = vulkan_ctx.log_device.get_queue(vkb::QueueType::graphics);
		if (!qr)
			throw std::runtime_error("no graphics queue");
		vulkan_ctx.graphics_queue = qr.value();
		auto fr                   = vulkan_ctx.log_device.get_queue_index(vkb::QueueType::graphics);
		if (!fr)
			throw std::runtime_error("no graphics queue family");
		vulkan_ctx.graphics_queue_family = fr.value();
	}

	// ==========================
	// === IV. VMA
	// ==========================
	{
		VmaVulkanFunctions vf{};
		vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
		vf.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
		VmaAllocatorCreateInfo ai{};
		ai.instance         = vulkan_ctx.instance.instance;
		ai.physicalDevice   = vulkan_ctx.phys_device.physical_device;
		ai.device           = vulkan_ctx.device;
		ai.vulkanApiVersion = VK_API_VERSION_1_3;
		ai.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
		ai.pVulkanFunctions = &vf;
		if (vmaCreateAllocator(&ai, &render_target_ctx.allocator) != VK_SUCCESS)
			throw std::runtime_error("failed to create VMA allocator");
		std::cout << "[INFO] VMA allocator created\n";
	}
	VmaAllocator allocator = render_target_ctx.allocator;

	// ===================================
	// === V. Swapchain
	// ===================================
	{
		auto r =
		    vkb::SwapchainBuilder{vulkan_ctx.log_device}
		        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
		        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		        .set_desired_extent(m_window->width(), m_window->height())
		        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
		        .build();
		if (!r)
			throw std::runtime_error("failed to create swapchain");
		swapchain_ctx.swapchain    = r.value();
		swapchain_ctx.image_format = swapchain_ctx.swapchain.image_format;
		auto ir                    = swapchain_ctx.swapchain.get_images();
		if (!ir)
			throw std::runtime_error("no swapchain images");
		swapchain_ctx.images = ir.value();
		auto vr              = swapchain_ctx.swapchain.get_image_views();
		if (!vr)
			throw std::runtime_error("no swapchain image views");
		swapchain_ctx.image_views = vr.value();
		swapchain_ctx.image_initialized.resize(swapchain_ctx.images.size(), false);
		std::cout << "[INFO] Swapchain: " << swapchain_ctx.images.size() << " images\n";
	}

	// ========================================
	// === VI. Render target images
	// ========================================
	{
		VkExtent3D ext          = {(uint32_t) m_window->width(), (uint32_t) m_window->height(), 1};
		auto       make_storage = [&](VkImage& img, VmaAllocation& a, VkImageUsageFlags extra) {
            VkImageCreateInfo ii{};
            ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ii.imageType     = VK_IMAGE_TYPE_2D;
            ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
            ii.extent        = ext;
            ii.mipLevels     = 1;
            ii.arrayLayers   = 1;
            ii.samples       = VK_SAMPLE_COUNT_1_BIT;
            ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | extra;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            if (vmaCreateImage(allocator, &ii, &ai, &img, &a, nullptr) != VK_SUCCESS)
                throw std::runtime_error("failed to create storage image");
		};
		auto make_dummy = [&](VkImageType itype, VkExtent3D e, VkImage& img, VmaAllocation& a) {
			VkImageCreateInfo ii{};
			ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			ii.imageType     = itype;
			ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
			ii.extent        = e;
			ii.mipLevels     = 1;
			ii.arrayLayers   = 1;
			ii.samples       = VK_SAMPLE_COUNT_1_BIT;
			ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
			ii.usage         = VK_IMAGE_USAGE_SAMPLED_BIT;
			ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			VmaAllocationCreateInfo ai{};
			ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			if (vmaCreateImage(allocator, &ii, &ai, &img, &a, nullptr) != VK_SUCCESS)
				throw std::runtime_error("failed to create dummy image");
		};
		make_storage(render_target_ctx.storage_image, render_target_ctx.storage_image_alloc,
		             VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc, 0);
		render_target_ctx.storage_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.storage_image,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.accum_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.accum_image,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		make_dummy(VK_IMAGE_TYPE_2D, {1, 1, 1}, render_target_ctx.dummy_image_2d,
		           render_target_ctx.dummy_image_2d_alloc);
		render_target_ctx.dummy_image_2d_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.dummy_image_2d,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		make_dummy(VK_IMAGE_TYPE_3D, {1, 1, 1}, render_target_ctx.dummy_image_3d,
		           render_target_ctx.dummy_image_3d_alloc);
		render_target_ctx.dummy_image_3d_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.dummy_image_3d,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_3D);
		std::cout << "[INFO] Render targets " << ext.width << "x" << ext.height << "\n";
	}

	// =============================================
	// === VI.5  Command pool (needed early)
	// =============================================
	{
		VkCommandPoolCreateInfo pi{};
		pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pi.queueFamilyIndex = vulkan_ctx.graphics_queue_family;
		pi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if (vkCreateCommandPool(vulkan_ctx.device, &pi, nullptr, &command_ctx.command_pool) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create command pool");
		std::cout << "[INFO] Command pool created\n";
	}
	// Transition dummy images → SHADER_READ_ONLY once
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
		for (VkImage img : {render_target_ctx.dummy_image_2d, render_target_ctx.dummy_image_3d})
			transition_layout(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED,
			                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
			                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
	}

	// =================================
	// === VII. Upload scene buffers
	// =================================
	{
		constexpr VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		GPUCamera gc = m_scene->m_camera.pack();
		scene_ctx.camera_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUCamera), &gc, SB, scene_ctx.camera_alloc);
		VmaAllocationCreateInfo cam_alloc_info{};
		cam_alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
		cam_alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBufferCreateInfo cam_buf_info{};
		cam_buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		cam_buf_info.size  = sizeof(GPUCamera);
		cam_buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationInfo cam_info;
		vmaCreateBuffer(allocator, &cam_buf_info, &cam_alloc_info, &scene_ctx.camera_buffer,
		                &scene_ctx.camera_alloc, &cam_info);
		scene_ctx.camera_mapped = cam_info.pMappedData;
		GPUCamera initial_cam   = m_scene->m_camera.pack();
		memcpy(scene_ctx.camera_mapped, &initial_cam, sizeof(GPUCamera));
		std::vector<GPUMaterial> gms;

		for (auto& mesh : m_scene->m_meshes)
			gms.push_back(mesh->m_material->pack());
		scene_ctx.material_count  = (uint32_t) gms.size();
		scene_ctx.material_buffer = create_and_upload_buffer(
		    allocator, sizeof(GPUMaterial) * gms.size(), gms.data(), SB, scene_ctx.material_alloc);
		std::cout << "[INFO] Uploaded " << scene_ctx.material_count << " materials\n";
	}

	// ======================================================
	// === VIII. BLAS per mesh + TLAS
	// ======================================================
	std::vector<GPUVertex> all_verts;
	std::vector<uint32_t>  all_idxs;
	std::vector<GPUMesh>   gpu_meshes;

	for (int mi = 0; mi < (int) m_scene->m_meshes.size(); ++mi) {
		auto& mesh = m_scene->m_meshes[mi];
		if (!mesh->m_material) {
			std::cerr << "[WARN] mesh " << mi << " no material\n";
			continue;
		}
		int voff = (int) all_verts.size(), ioff = (int) all_idxs.size();
		all_verts.insert(all_verts.end(), mesh->gpuVertices.begin(), mesh->gpuVertices.end());
		all_idxs.insert(all_idxs.end(), mesh->gpuIndices.begin(), mesh->gpuIndices.end());
		BLAS    blas     = build_blas(allocator, vulkan_ctx.device, command_ctx.command_pool,
		                              vulkan_ctx.graphics_queue, mesh->gpuVertices.data(),
		                              (uint32_t) mesh->gpuVertices.size(), sizeof(GPUVertex),
		                              mesh->gpuIndices.data(), (uint32_t) mesh->gpuIndices.size());
		GPUMesh gm       = mesh->pack(mi, voff, ioff);
		gm.blasHandle_lo = (uint32_t) (blas.device_address & 0xFFFFFFFF);
		gm.blasHandle_hi = (uint32_t) (blas.device_address >> 32);
		as_ctx.blases.push_back(std::move(blas));
		gpu_meshes.push_back(gm);
		std::cout << "[INFO] BLAS " << mi << ": " << mesh->gpuIndices.size() / 3 << " tris\n";
	}
	if (!as_ctx.blases.empty()) {
		build_tlas(allocator, vulkan_ctx.device, command_ctx.command_pool,
		           vulkan_ctx.graphics_queue, as_ctx.blases, m_scene->m_meshes);
		std::cout << "[INFO] TLAS: " << as_ctx.blases.size() << " instances\n";
	}
	if (!gpu_meshes.empty()) {
		constexpr VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		scene_ctx.mesh_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUMesh) * gpu_meshes.size(),
		                             gpu_meshes.data(), SB, scene_ctx.mesh_alloc);
		scene_ctx.vertex_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUVertex) * all_verts.size(),
		                             all_verts.data(), SB, scene_ctx.vertex_alloc);
		scene_ctx.index_buffer =
		    create_and_upload_buffer(allocator, sizeof(uint32_t) * all_idxs.size(), all_idxs.data(),
		                             SB, scene_ctx.index_alloc);
		scene_ctx.mesh_count = (uint32_t) gpu_meshes.size();
		std::cout << "[INFO] Mesh buffers: " << gpu_meshes.size() << " meshes, " << all_verts.size()
		          << " verts, " << all_idxs.size() << " idxs\n";
	}

	// ===================================================
	// === VIII.5  Upload OpenPBR LUTs + scene textures
	// ===================================================
	upload_openpbr_luts(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                    vulkan_ctx.graphics_queue);
	upload_material_textures(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                         vulkan_ctx.graphics_queue, m_scene->m_textures);

	// ==============================================
	// === IX. Descriptor layout, pool, sets
	// ==============================================
	// Binding map:
	//   0  = output image            STORAGE_IMAGE
	//   1  = accum image             STORAGE_IMAGE
	//   2  = camera buffer           STORAGE_BUFFER
	//   3  = shared sampler          SAMPLER (LUTs + material textures)
	//   4  = materials buffer        STORAGE_BUFFER
	//   5  = TLAS                    ACCELERATION_STRUCTURE
	//   6  = meshes buffer           STORAGE_BUFFER
	//   7  = vertices buffer         STORAGE_BUFFER
	//   8  = indices buffer          STORAGE_BUFFER
	//   9  = lut_textures_2d[8]      SAMPLED_IMAGE × NUM_LUTS
	//  10  = lut_textures_3d[8]      SAMPLED_IMAGE × NUM_LUTS
	//  11  = material_textures[256]  SAMPLED_IMAGE × MAX_MATERIAL_TEXTURES
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
		    make_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(3, VK_DESCRIPTOR_TYPE_SAMPLER),
		    make_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, NUM_LUTS),
		    make_binding(10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, NUM_LUTS),
		    make_binding(11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_MATERIAL_TEXTURES),
		};
		if (as_ctx.tlas != VK_NULL_HANDLE)
			bindings.push_back(make_binding(5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR));
		VkDescriptorSetLayoutCreateInfo dsl{};
		dsl.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dsl.bindingCount = (uint32_t) bindings.size();
		dsl.pBindings    = bindings.data();
		if (vkCreateDescriptorSetLayout(vulkan_ctx.device, &dsl, nullptr,
		                                &render_target_ctx.descriptor_set_layout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout");
		std::cout << "[INFO] Descriptor set layout created\n";
	}
	{
		std::vector<VkDescriptorPoolSize> ps = {
		    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
		    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5},
		    {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
		    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * NUM_LUTS + MAX_MATERIAL_TEXTURES},
		};
		if (as_ctx.tlas != VK_NULL_HANDLE)
			ps.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1});
		VkDescriptorPoolCreateInfo pi{};
		pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pi.maxSets       = 1;
		pi.poolSizeCount = (uint32_t) ps.size();
		pi.pPoolSizes    = ps.data();
		if (vkCreateDescriptorPool(vulkan_ctx.device, &pi, nullptr,
		                           &render_target_ctx.descriptor_pool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool");
		VkDescriptorSetAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool     = render_target_ctx.descriptor_pool;
		ai.descriptorSetCount = 1;
		ai.pSetLayouts        = &render_target_ctx.descriptor_set_layout;
		if (vkAllocateDescriptorSets(vulkan_ctx.device, &ai, &render_target_ctx.descriptor_set) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to allocate descriptor set");
		std::cout << "[INFO] Descriptor pool+set allocated\n";
	}
	// Shared linear-clamp sampler
	{
		VkSamplerCreateInfo si{};
		si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		si.magFilter    = VK_FILTER_LINEAR;
		si.minFilter    = VK_FILTER_LINEAR;
		si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		si.maxLod                                           = VK_LOD_CLAMP_NONE;
		if (vkCreateSampler(vulkan_ctx.device, &si, nullptr, &render_target_ctx.sampler) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create sampler");
	}
	// Write descriptors
	{
		std::vector<VkWriteDescriptorSet> writes;
		// 0 — output image
		VkDescriptorImageInfo oi{};
		oi.imageView   = render_target_ctx.storage_image_view;
		oi.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 0, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &oi});
		// 1 — accum image
		VkDescriptorImageInfo ai2{};
		ai2.imageView   = render_target_ctx.accum_image_view;
		ai2.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 1, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ai2});
		// 2 — camera
		VkDescriptorBufferInfo cb{scene_ctx.camera_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 2, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &cb});
		// 3 — sampler
		VkDescriptorImageInfo si2{};
		si2.sampler = render_target_ctx.sampler;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER,
		                  &si2});
		// 4 — materials
		VkDescriptorBufferInfo mb{scene_ctx.material_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 4, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mb});
		// 5 — TLAS
		VkWriteDescriptorSetAccelerationStructureKHR as_ext{};
		VkWriteDescriptorSet                         as_write{};
		if (as_ctx.tlas != VK_NULL_HANDLE) {
			as_ext.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
			as_ext.accelerationStructureCount = 1;
			as_ext.pAccelerationStructures    = &as_ctx.tlas;
			as_write                          = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			                                     &as_ext,
			                                     render_target_ctx.descriptor_set,
			                                     5,
			                                     0,
			                                     1,
			                                     VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR};
			writes.push_back(as_write);
		}
		// 6 — meshes
		VkDescriptorBufferInfo mshb{scene_ctx.mesh_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 6, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mshb});
		// 7 — vertices
		VkDescriptorBufferInfo vb{scene_ctx.vertex_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 7, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vb});
		// 8 — indices
		VkDescriptorBufferInfo ib{scene_ctx.index_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 8, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ib});
		// 9 — LUT 2D (real or dummy fallback per slot)
		std::vector<VkDescriptorImageInfo> lut2(NUM_LUTS);
		for (uint32_t i = 0; i < NUM_LUTS; ++i) {
			lut2[i].imageView   = (render_target_ctx.lut_textures_2d[i].view != VK_NULL_HANDLE) ?
			                          render_target_ctx.lut_textures_2d[i].view :
			                          render_target_ctx.dummy_image_2d_view;
			lut2[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		{
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = 9;
			w.dstArrayElement = 0;
			w.descriptorCount = NUM_LUTS;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			w.pImageInfo      = lut2.data();
			writes.push_back(w);
		}
		// 10 — LUT 3D
		std::vector<VkDescriptorImageInfo> lut3(NUM_LUTS);
		for (uint32_t i = 0; i < NUM_LUTS; ++i) {
			lut3[i].imageView   = (render_target_ctx.lut_textures_3d[i].view != VK_NULL_HANDLE) ?
			                          render_target_ctx.lut_textures_3d[i].view :
			                          render_target_ctx.dummy_image_3d_view;
			lut3[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		{
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = 10;
			w.dstArrayElement = 0;
			w.descriptorCount = NUM_LUTS;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			w.pImageInfo      = lut3.data();
			writes.push_back(w);
		}
		// 11 — Material textures (pad unused slots with dummy)
		std::vector<VkDescriptorImageInfo> mt(MAX_MATERIAL_TEXTURES);
		for (uint32_t i = 0; i < MAX_MATERIAL_TEXTURES; ++i) {
			bool ok = i < render_target_ctx.mat_views.size() &&
			          render_target_ctx.mat_views[i] != VK_NULL_HANDLE;
			mt[i].imageView =
			    ok ? render_target_ctx.mat_views[i] : render_target_ctx.dummy_image_2d_view;
			mt[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		{
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = 11;
			w.dstArrayElement = 0;
			w.descriptorCount = MAX_MATERIAL_TEXTURES;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			w.pImageInfo      = mt.data();
			writes.push_back(w);
		}
		vkUpdateDescriptorSets(vulkan_ctx.device, (uint32_t) writes.size(), writes.data(), 0,
		                       nullptr);
		std::cout << "[INFO] Descriptor sets updated\n";
	}

	// ============================================
	// === X. Compute Pipeline
	// ============================================
	{
		auto spirv = compile_slang_shader(std::string(SHADERS_DIR) + "/pathtracer.slang", "main",
		                                  {VENDORS_DIR});
		std::cout << "[INFO] Shader: " << spirv.size() * 4 << " bytes SPIR-V\n";
		VkShaderModuleCreateInfo mci{};
		mci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		mci.codeSize = spirv.size() * sizeof(uint32_t);
		mci.pCode    = spirv.data();
		VkShaderModule sm;
		if (vkCreateShaderModule(vulkan_ctx.device, &mci, nullptr, &sm) != VK_SUCCESS)
			throw std::runtime_error("failed to create shader module");
		VkPushConstantRange pr{};
		pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pr.size       = sizeof(uint32_t) * 2;
		VkPipelineLayoutCreateInfo pli{};
		pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount         = 1;
		pli.pSetLayouts            = &render_target_ctx.descriptor_set_layout;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges    = &pr;
		if (vkCreatePipelineLayout(vulkan_ctx.device, &pli, nullptr,
		                           &compute_ctx.pipeline_layout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout");
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = sm;
		stage.pName  = "main";
		VkComputePipelineCreateInfo pci{};
		pci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pci.stage  = stage;
		pci.layout = compute_ctx.pipeline_layout;
		if (vkCreateComputePipelines(vulkan_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr,
		                             &compute_ctx.pipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create compute pipeline");
		vkDestroyShaderModule(vulkan_ctx.device, sm, nullptr);
		std::cout << "[INFO] Compute pipeline created\n";
	}

	// ================================================
	// === XI. Main command buffer
	// ================================================
	{
		VkCommandBufferAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ai.commandPool        = command_ctx.command_pool;
		ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = 1;
		if (vkAllocateCommandBuffers(vulkan_ctx.device, &ai, &command_ctx.command_buffer) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to allocate command buffer");
	}
	// Accum image → GENERAL
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
		transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
		std::cout << "[INFO] Accum image → GENERAL\n";
	}

	// ==========================
	// === XII. Sync objects
	// ==========================
	{
		VkSemaphoreCreateInfo sei{};
		sei.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkFenceCreateInfo fi{};
		fi.sType         = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fi.flags         = VK_FENCE_CREATE_SIGNALED_BIT;
		const uint32_t n = (uint32_t) swapchain_ctx.images.size();
		sync_ctx.image_available.resize(n);
		sync_ctx.render_finished.resize(n);
		for (uint32_t i = 0; i < n; ++i) {
			if (vkCreateSemaphore(vulkan_ctx.device, &sei, nullptr, &sync_ctx.image_available[i]) !=
			    VK_SUCCESS)
				throw std::runtime_error("failed to create semaphore");
			if (vkCreateSemaphore(vulkan_ctx.device, &sei, nullptr, &sync_ctx.render_finished[i]) !=
			    VK_SUCCESS)
				throw std::runtime_error("failed to create semaphore");
		}
		if (vkCreateFence(vulkan_ctx.device, &fi, nullptr, &sync_ctx.in_flight) != VK_SUCCESS)
			throw std::runtime_error("failed to create fence");
		std::cout << "[INFO] Sync objects created\n";
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

	// Initialise fly camera from the scene camera
	FlyCamera fly_cam(m_scene->m_camera.m_position, m_scene->m_camera.m_target,
	                  m_scene->m_camera.m_fov, 0.5f);

	double last_time = glfwGetTime();

	// One-shot key-press trackers
	int prev_f6  = GLFW_RELEASE;
	int prev_f11 = GLFW_RELEASE;

	while (!m_window->shouldClose()) {
		glfwPollEvents();

		double now = glfwGetTime();
		float  dt  = static_cast<float>(now - last_time);
		last_time  = now;
		dt         = std::clamp(dt, 0.0001f, 0.1f);        // guard against huge dt on freeze

		// ---- Check if the window was resized --------------------------------
		uint32_t fb_w = m_window->width();
		uint32_t fb_h = m_window->height();
		if (fb_w != swapchain_ctx.swapchain.extent.width ||
		    fb_h != swapchain_ctx.swapchain.extent.height) {
			handle_resize(frame_number, fb_w, fb_h);
		}

		// ---- F11: fullscreen toggle -----------------------------------------
		int f11 = glfwGetKey(m_window->handle(), GLFW_KEY_F11);
		if (f11 == GLFW_PRESS && prev_f11 == GLFW_RELEASE) {
			m_window->toggle_fullscreen();
			// Resize detected next frame automatically
		}
		prev_f11 = f11;

		// ---- F6: shader hot-reload ------------------------------------------
		int f6 = glfwGetKey(m_window->handle(), GLFW_KEY_F6);
		if (f6 == GLFW_PRESS && prev_f6 == GLFW_RELEASE) {
			if (rebuild_pipeline())
				frame_number = 0;        // reset accumulation on success
		}
		prev_f6 = f6;

		// ---- Fly-camera update ----------------------------------------------
		if (fly_cam.update(m_window->handle(), dt))
			frame_number = 0;        // camera moved → reset accumulation

		// Upload camera to GPU (persistent mapping – no staging needed)
		{
			GPUCamera gpu_cam = fly_cam.pack();
			memcpy(scene_ctx.camera_mapped, &gpu_cam, sizeof(GPUCamera));
		}

		// ---- Frame rendering ------------------------------------------------
		const uint32_t frame_idx = frame_number % (uint32_t) swapchain_ctx.images.size();

		vkWaitForFences(vulkan_ctx.device, 1, &sync_ctx.in_flight, VK_TRUE, UINT64_MAX);
		vkResetFences(vulkan_ctx.device, 1, &sync_ctx.in_flight);

		uint32_t image_index;
		VkResult acquire_result = vkAcquireNextImageKHR(
		    vulkan_ctx.device, swapchain_ctx.swapchain.swapchain, UINT64_MAX,
		    sync_ctx.image_available[frame_idx], VK_NULL_HANDLE, &image_index);
		if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
			handle_resize(frame_number, fb_w, fb_h);
			continue;
		}

		vkResetCommandBuffer(command_ctx.command_buffer, 0);
		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(command_ctx.command_buffer, &begin_info);
		VkCommandBuffer cmd = command_ctx.command_buffer;

		VkImageLayout storage_old = render_target_ctx.storage_image_initialized ?
		                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
		                                VK_IMAGE_LAYOUT_UNDEFINED;
		transition_layout(cmd, render_target_ctx.storage_image, storage_old,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		VkImageLayout swap_old = swapchain_ctx.image_initialized[image_index] ?
		                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
		                             VK_IMAGE_LAYOUT_UNDEFINED;
		transition_layout(cmd, swapchain_ctx.images[image_index], swap_old,
		                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline_layout, 0,
		                        1, &render_target_ctx.descriptor_set, 0, nullptr);

		struct PC {
			uint32_t frame, material_count;
		};
		PC pc{frame_number, scene_ctx.material_count};
		vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
		                   sizeof(pc), &pc);

		vkCmdDispatch(cmd, (m_window->width() + 15) / 16, (m_window->height() + 15) / 16, 1);

		render_target_ctx.storage_image_initialized  = true;
		swapchain_ctx.image_initialized[image_index] = true;
		frame_number++;

		transition_layout(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_GENERAL,
		                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                  VK_PIPELINE_STAGE_TRANSFER_BIT);

		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.srcOffsets[1]  = {(int32_t) m_window->width(), (int32_t) m_window->height(), 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.dstOffsets[1]  = {(int32_t) swapchain_ctx.swapchain.extent.width,
		                       (int32_t) swapchain_ctx.swapchain.extent.height, 1};
		vkCmdBlitImage(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
		               &blit, VK_FILTER_NEAREST);

		transition_layout(cmd, swapchain_ctx.images[image_index],
		                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		                  VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

		vkEndCommandBuffer(cmd);

		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		VkSubmitInfo         submit{};
		submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.waitSemaphoreCount   = 1;
		submit.pWaitSemaphores      = &sync_ctx.image_available[frame_idx];
		submit.pWaitDstStageMask    = &wait_stage;
		submit.commandBufferCount   = 1;
		submit.pCommandBuffers      = &cmd;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores    = &sync_ctx.render_finished[image_index];
		vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &submit, sync_ctx.in_flight);

		VkPresentInfoKHR present{};
		present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present.waitSemaphoreCount = 1;
		present.pWaitSemaphores    = &sync_ctx.render_finished[image_index];
		present.swapchainCount     = 1;
		present.pSwapchains        = &swapchain_ctx.swapchain.swapchain;
		present.pImageIndices      = &image_index;
		VkResult present_result    = vkQueuePresentKHR(vulkan_ctx.graphics_queue, &present);
		if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
			handle_resize(frame_number, fb_w, fb_h);
		}
	}

	vkDeviceWaitIdle(vulkan_ctx.device);
}