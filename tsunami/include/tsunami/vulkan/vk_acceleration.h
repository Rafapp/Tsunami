#pragma once

#include <memory>
#include <vector>

#include "tsunami/vulkan/internal/vk_context.h"

class Mesh;

BLAS build_blas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q, const void* verts,
                uint32_t vc, uint32_t vstride, const void* idxs, uint32_t ic);

void build_tlas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                const std::vector<BLAS>& blases, const std::vector<std::unique_ptr<Mesh>>& meshes);
