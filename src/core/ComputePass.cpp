#include "ComputePass.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

void ComputePass::init(VkDevice device) {
  createDescriptorSetLayout(device);
  createDescriptorPool(device);
  allocateDescriptorSets(device);
  createPipelineLayout(device);
  createPipeline(device);
}

void ComputePass::cleanup(VkDevice device) {
  vkDestroyPipeline(device, pipeline_, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
  vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout_, nullptr);
}

void ComputePass::createDescriptorSetLayout(VkDevice device) {
  VkDescriptorSetLayoutBinding b[4]{};
  b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
          nullptr};
  b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
          nullptr};
  b[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
          nullptr};
  b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
          nullptr};

  VkDescriptorSetLayoutCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = 4;
  ci.pBindings = b;
  if (vkCreateDescriptorSetLayout(device, &ci, nullptr,
                                  &descriptorSetLayout_) != VK_SUCCESS)
    throw std::runtime_error(
        "ComputePass: failed to create descriptor set layout.");
}

void ComputePass::createDescriptorPool(VkDevice device) {
  VkDescriptorPoolSize sizes[3]{};
  sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kMaxFramesInFlight};
  sizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxFramesInFlight};
  sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxFramesInFlight};

  VkDescriptorPoolCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  ci.poolSizeCount = 3;
  ci.pPoolSizes = sizes;
  ci.maxSets = kMaxFramesInFlight;
  if (vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool_) !=
      VK_SUCCESS)
    throw std::runtime_error("ComputePass: failed to create descriptor pool.");
}

void ComputePass::allocateDescriptorSets(VkDevice device) {
  VkDescriptorSetLayout layouts[kMaxFramesInFlight];
  for (int i = 0; i < kMaxFramesInFlight; ++i)
    layouts[i] = descriptorSetLayout_;

  VkDescriptorSetAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = descriptorPool_;
  ai.descriptorSetCount = kMaxFramesInFlight;
  ai.pSetLayouts = layouts;
  if (vkAllocateDescriptorSets(device, &ai, descriptorSets_.data()) !=
      VK_SUCCESS)
    throw std::runtime_error(
        "ComputePass: failed to allocate descriptor sets.");
}

void ComputePass::createPipelineLayout(VkDevice device) {
  VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};

  VkPipelineLayoutCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ci.setLayoutCount = 1;
  ci.pSetLayouts = &descriptorSetLayout_;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &pc;
  if (vkCreatePipelineLayout(device, &ci, nullptr, &pipelineLayout_) !=
      VK_SUCCESS)
    throw std::runtime_error("ComputePass: failed to create pipeline layout.");
}

void ComputePass::createPipeline(VkDevice device) {
  auto code = readSpv("shaders/raymarch.comp.spv");
  auto module = createShaderModule(device, code);

  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";

  VkComputePipelineCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  ci.stage = stage;
  ci.layout = pipelineLayout_;

  if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr,
                               &pipeline_) != VK_SUCCESS) {
    vkDestroyShaderModule(device, module, nullptr);
    throw std::runtime_error("ComputePass: failed to create compute pipeline.");
  }
  vkDestroyShaderModule(device, module, nullptr);
}

// ── Hot reload
// ────────────────────────────────────────────────────────────────

bool ComputePass::reloadPipeline(VkDevice device) {
  std::vector<char> code;
  try {
    code = readSpv("shaders/raymarch.comp.spv");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[ComputePass] Reload failed (read): %s\n", e.what());
    return false;
  }

  VkShaderModule module;
  try {
    module = createShaderModule(device, code);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[ComputePass] Reload failed (module): %s\n",
                 e.what());
    return false;
  }

  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";

  VkComputePipelineCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  ci.stage = stage;
  ci.layout = pipelineLayout_;

  VkPipeline newPipeline;
  VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci,
                                             nullptr, &newPipeline);
  vkDestroyShaderModule(device, module, nullptr);

  if (result != VK_SUCCESS) {
    std::fprintf(stderr, "[ComputePass] Reload failed (pipeline %d)\n", result);
    return false;
  }

  vkDestroyPipeline(device, pipeline_, nullptr);
  pipeline_ = newPipeline;
  std::fprintf(stderr, "[ComputePass] Pipeline reloaded.\n");
  return true;
}

// ── Descriptor writes
// ─────────────────────────────────────────────────────────

void ComputePass::writeDescriptors(
    VkDevice device, VkBuffer nodeBuffer, VkDeviceSize nodeBytes,
    VkBuffer materialBuffer, VkDeviceSize materialBytes,
    const std::array<VkBuffer, kMaxFramesInFlight> &uboBuffers,
    VkImageView storageImageView) {
  for (int f = 0; f < kMaxFramesInFlight; ++f) {
    VkDescriptorBufferInfo nodeBI{nodeBuffer, 0, nodeBytes};
    VkDescriptorBufferInfo matBI{materialBuffer, 0, materialBytes};
    VkDescriptorBufferInfo uboBI{uboBuffers[f], 0, sizeof(gpu::SceneUBO)};
    VkDescriptorImageInfo imgInfo{VK_NULL_HANDLE, storageImageView,
                                  VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[4]{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 descriptorSets_[f],
                 0,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 nullptr,
                 &nodeBI,
                 nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 descriptorSets_[f],
                 1,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 nullptr,
                 &matBI,
                 nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 descriptorSets_[f],
                 2,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 nullptr,
                 &uboBI,
                 nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 descriptorSets_[f],
                 3,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                 &imgInfo,
                 nullptr,
                 nullptr};

    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  }
}

void ComputePass::updateStorageImage(VkDevice device,
                                     VkImageView storageImageView) {
  VkDescriptorImageInfo imgInfo{VK_NULL_HANDLE, storageImageView,
                                VK_IMAGE_LAYOUT_GENERAL};
  for (int f = 0; f < kMaxFramesInFlight; ++f) {
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                           nullptr,
                           descriptorSets_[f],
                           3,
                           0,
                           1,
                           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                           &imgInfo,
                           nullptr,
                           nullptr};
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
  }
}

void ComputePass::rebindBuffers(VkDevice device, VkBuffer nodeBuffer,
                                VkDeviceSize nodeBytes, VkBuffer materialBuffer,
                                VkDeviceSize materialBytes) {
  for (int f = 0; f < kMaxFramesInFlight; ++f) {
    VkDescriptorBufferInfo nodeBI{nodeBuffer, 0, nodeBytes};
    VkDescriptorBufferInfo matBI{materialBuffer, 0, materialBytes};

    VkWriteDescriptorSet writes[2]{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 descriptorSets_[f],
                 0,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 nullptr,
                 &nodeBI,
                 nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 descriptorSets_[f],
                 1,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 nullptr,
                 &matBI,
                 nullptr};
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  }
}

// ── Dispatch
// ──────────────────────────────────────────────────────────────────

void ComputePass::dispatch(VkCommandBuffer cmd, int frameIndex,
                           VkExtent2D extent, int32_t bvhRootIndex) const {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                          0, 1, &descriptorSets_[frameIndex], 0, nullptr);
  PushConstants pc{bvhRootIndex};
  vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(pc), &pc);
  uint32_t gx = (extent.width + kWorkgroupSize - 1) / kWorkgroupSize;
  uint32_t gy = (extent.height + kWorkgroupSize - 1) / kWorkgroupSize;
  vkCmdDispatch(cmd, gx, gy, 1);
}

// ── Helpers
// ───────────────────────────────────────────────────────────────────

VkShaderModule ComputePass::createShaderModule(VkDevice device,
                                               const std::vector<char> &code) {
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module;
  if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
    throw std::runtime_error("ComputePass: failed to create shader module.");
  return module;
}

std::vector<char> ComputePass::readSpv(const std::string &path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("ComputePass: failed to open SPIR-V file: " +
                             path);
  size_t size = static_cast<size_t>(file.tellg());
  std::vector<char> buf(size);
  file.seekg(0);
  file.read(buf.data(), static_cast<std::streamsize>(size));
  return buf;
}
