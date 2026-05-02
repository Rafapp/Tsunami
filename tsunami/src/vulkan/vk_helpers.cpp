#include "tsunami/vulkan/vk_helpers.h"

#include <iostream>
#include <stdexcept>

void check_vk_result(VkResult result) {
	if (result == VK_SUCCESS)
		return;
	std::cerr << "[Vulkan] VkResult = " << result << "\n";
	if (result < 0)
		throw std::runtime_error("Vulkan call failed");
}

VkBuffer create_gpu_buffer(VmaAllocator alloc, VkDeviceSize sz, VkBufferUsageFlags usage,
                           VmaAllocation& out, VkDeviceSize align) {
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

VkBuffer create_and_upload_buffer(VmaAllocator alloc, VkDeviceSize sz, const void* data,
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

VkDeviceAddress get_bda(VkDevice dev, VkBuffer buf) {
	VkBufferDeviceAddressInfo i{};
	i.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	i.buffer = buf;
	return vkGetBufferDeviceAddress(dev, &i);
}

VkImageView create_image_view(VkDevice dev, VkImage img, VkFormat fmt, VkImageViewType type,
                              VkImageAspectFlags asp) {
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

void transition_layout(VkCommandBuffer cmd, VkImage img, VkImageLayout old_layout,
                       VkImageLayout new_layout, VkAccessFlags src, VkAccessFlags dst,
                       VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
	VkImageMemoryBarrier b{};
	b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	b.oldLayout           = old_layout;
	b.newLayout           = new_layout;
	b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image                                       = img;
	b.subresourceRange                            = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	b.srcAccessMask                               = src;
	b.dstAccessMask                               = dst;
	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

VkCommandBuffer begin_one_time_cmd(VkDevice dev, VkCommandPool pool) {
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

void end_one_time_cmd(VkDevice dev, VkCommandPool pool, VkQueue q, VkCommandBuffer cmd) {
	vkEndCommandBuffer(cmd);
	VkSubmitInfo si{};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cmd;
	vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(q);
	vkFreeCommandBuffers(dev, pool, 1, &cmd);
}
