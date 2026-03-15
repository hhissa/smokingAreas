#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../gpu/gpuTypes.h"

// ─────────────────────────────────────────────────────────────────────────────
// ComputePass
//
// Descriptor set layout (set 0):
//   binding 0 — STORAGE_BUFFER  NodeBuffer      (GPUNode[])
//   binding 1 — STORAGE_BUFFER  MaterialBuffer  (GPUMaterial[])
//   binding 2 — UNIFORM_BUFFER  SceneUBO        (camera, time, resolution)
//   binding 3 — STORAGE_IMAGE   outputImage     (rgba8, writeonly)
//
// One descriptor set per frame-in-flight so the UBO slot for frame N and N+1
// can differ without racing.
//
// Push constants (4 bytes):
//   int bvhRootIndex — index of BVH root in the flat GPUNode array.
//
// Buffer ownership:
//   ComputePass does NOT own the VkBuffers — Application does.
//   ComputePass only holds descriptor writes pointing at them.
//   Call rebindBuffers() whenever the node/material buffer is recreated.
// ─────────────────────────────────────────────────────────────────────────────

class ComputePass {
public:
  static constexpr uint32_t kWorkgroupSize = 8;
  static constexpr int kMaxFramesInFlight = 2;

  struct PushConstants {
    int32_t bvhRootIndex; // 4 bytes
  };

  // Create layout, pool, sets, pipeline.
  // Buffers don't exist yet — call writeDescriptors() once they do.
  void init(VkDevice device);
  void cleanup(VkDevice device);

  // Bind all descriptors.  Call once after init() when all buffers are ready.
  void
  writeDescriptors(VkDevice device, VkBuffer nodeBuffer, VkDeviceSize nodeBytes,
                   VkBuffer materialBuffer, VkDeviceSize materialBytes,
                   const std::array<VkBuffer, kMaxFramesInFlight> &uboBuffers,
                   VkImageView storageImageView);

  // Rebind storage image after resize.
  void updateStorageImage(VkDevice device, VkImageView storageImageView);

  // Rebind SSBO descriptors after scene rebuild caused buffer growth.
  void rebindBuffers(VkDevice device, VkBuffer nodeBuffer,
                     VkDeviceSize nodeBytes, VkBuffer materialBuffer,
                     VkDeviceSize materialBytes);

  void dispatch(VkCommandBuffer cmd, int frameIndex, VkExtent2D extent,
                int32_t bvhRootIndex) const;

private:
  VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, kMaxFramesInFlight> descriptorSets_ = {};
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;

  void createDescriptorSetLayout(VkDevice device);
  void createDescriptorPool(VkDevice device);
  void allocateDescriptorSets(VkDevice device);
  void createPipelineLayout(VkDevice device);
  void createPipeline(VkDevice device);

  static VkShaderModule createShaderModule(VkDevice device,
                                           const std::vector<char> &code);
  static std::vector<char> readSpv(const std::string &path);
};
;
