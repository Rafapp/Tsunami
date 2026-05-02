#pragma once

#include <memory>
#include <vector>

#include "tsunami/vulkan/internal/vk_context.h"

class Texture;

void upload_openpbr_luts(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q);

void upload_material_textures(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const std::vector<std::shared_ptr<Texture>>& textures);
