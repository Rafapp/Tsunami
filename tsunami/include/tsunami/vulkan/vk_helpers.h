#pragma once

#include "tsunami/vulkan/internal/vk_context.h"

void check_vk_result(VkResult result);

VkBuffer create_gpu_buffer(VmaAllocator alloc, VkDeviceSize sz, VkBufferUsageFlags usage,
                           VmaAllocation& out, VkDeviceSize align = 0);

VkBuffer create_and_upload_buffer(VmaAllocator alloc, VkDeviceSize sz, const void* data,
                                  VkBufferUsageFlags usage, VmaAllocation& out);

VkDeviceAddress get_bda(VkDevice dev, VkBuffer buf);

VkImageView create_image_view(VkDevice dev, VkImage img, VkFormat fmt, VkImageViewType type,
                              VkImageAspectFlags asp = VK_IMAGE_ASPECT_COLOR_BIT);

void transition_layout(VkCommandBuffer cmd, VkImage img, VkImageLayout old_layout,
                       VkImageLayout new_layout, VkAccessFlags src, VkAccessFlags dst,
                       VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);

VkCommandBuffer begin_one_time_cmd(VkDevice dev, VkCommandPool pool);
void            end_one_time_cmd(VkDevice dev, VkCommandPool pool, VkQueue q, VkCommandBuffer cmd);
