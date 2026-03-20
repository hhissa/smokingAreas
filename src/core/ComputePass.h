#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../gpu/gpuTypes.h"

class ComputePass {
public:
  static constexpr uint32_t kWorkgroupSize = 8;
  static constexpr int kMaxFramesInFlight = 2;

  struct PushConstants {
    int32_t bvhRootIndex;
  };

  void init(VkDevice device);
  void cleanup(VkDevice device);

  void
  writeDescriptors(VkDevice device, VkBuffer nodeBuffer, VkDeviceSize nodeBytes,
                   VkBuffer materialBuffer, VkDeviceSize materialBytes,
                   const std::array<VkBuffer, kMaxFramesInFlight> &uboBuffers,
                   VkImageView storageImageView);

  void updateStorageImage(VkDevice device, VkImageView storageImageView);

  void rebindBuffers(VkDevice device, VkBuffer nodeBuffer,
                     VkDeviceSize nodeBytes, VkBuffer materialBuffer,
                     VkDeviceSize materialBytes);

  // Hot-reload: swap only the VkPipeline from the .spv on disk.
  // Call vkDeviceWaitIdle before this. Returns true on success.
  bool reloadPipeline(VkDevice device);

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
