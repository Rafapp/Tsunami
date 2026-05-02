#include "tsunami/vulkan/vk_acceleration.h"

#include <stdexcept>

#include <glm/glm.hpp>

#include "tsunami/scene/scene.h"

#include "tsunami/vulkan/vk_helpers.h"

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

// Vulkan expects a 3x4 row-major matrix; GLM is column-major.
static VkTransformMatrixKHR glm_to_vk_transform(const glm::mat4& m) {
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

BLAS build_blas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q, const void* verts,
                uint32_t vc, uint32_t vstride, const void* idxs, uint32_t ic) {
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

	const uint32_t tc = ic / 3;

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

	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
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

void build_tlas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                const std::vector<BLAS>& blases, const std::vector<std::unique_ptr<Mesh>>& meshes) {
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
