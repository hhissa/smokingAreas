#include "./ComputePass.h"

#include <fstream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Init / cleanup
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor set layout
// ─────────────────────────────────────────────────────────────────────────────

void ComputePass::createDescriptorSetLayout(VkDevice device) {
  VkDescriptorSetLayoutBinding b[4]{};

  // 0: node SSBO
  b[0].binding = 0;
  b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b[0].descriptorCount = 1;
  b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  // 1: material SSBO
  b[1].binding = 1;
  b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b[1].descriptorCount = 1;
  b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  // 2: scene UBO
  b[2].binding = 2;
  b[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  b[2].descriptorCount = 1;
  b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  // 3: output storage image
  b[3].binding = 3;
  b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  b[3].descriptorCount = 1;
  b[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = 4;
  ci.pBindings = b;

  if (vkCreateDescriptorSetLayout(device, &ci, nullptr,
                                  &descriptorSetLayout_) != VK_SUCCESS)
    throw std::runtime_error(
        "ComputePass: failed to create descriptor set layout.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Pool + sets
// ─────────────────────────────────────────────────────────────────────────────

void ComputePass::createDescriptorPool(VkDevice device) {
  // kMaxFramesInFlight sets, each with 2 SSBOs + 1 UBO + 1 storage image.
  VkDescriptorPoolSize sizes[3]{};
  sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sizes[0].descriptorCount =
      2 * kMaxFramesInFlight; // node + material per frame
  sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sizes[1].descriptorCount = kMaxFramesInFlight;
  sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  sizes[2].descriptorCount = kMaxFramesInFlight;

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

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline layout
// ─────────────────────────────────────────────────────────────────────────────

void ComputePass::createPipelineLayout(VkDevice device) {
  VkPushConstantRange pc{};
  pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc.offset = 0;
  pc.size = sizeof(PushConstants);

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

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor writes
// ─────────────────────────────────────────────────────────────────────────────

void ComputePass::writeDescriptors(
    VkDevice device, VkBuffer nodeBuffer, VkDeviceSize nodeBytes,
    VkBuffer materialBuffer, VkDeviceSize materialBytes,
    const std::array<VkBuffer, kMaxFramesInFlight> &uboBuffers,
    VkImageView storageImageView) {
  for (int f = 0; f < kMaxFramesInFlight; ++f) {
    VkDescriptorBufferInfo nodeBI{};
    nodeBI.buffer = nodeBuffer;
    nodeBI.offset = 0;
    nodeBI.range = nodeBytes;

    VkDescriptorBufferInfo matBI{};
    matBI.buffer = materialBuffer;
    matBI.offset = 0;
    matBI.range = materialBytes;

    VkDescriptorBufferInfo uboBI{};
    uboBI.buffer = uboBuffers[f];
    uboBI.offset = 0;
    uboBI.range = sizeof(gpu::SceneUBO);

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView = storageImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[4]{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptorSets_[f];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &nodeBI;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSets_[f];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &matBI;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptorSets_[f];
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &uboBI;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = descriptorSets_[f];
    writes[3].dstBinding = 3;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].descriptorCount = 1;
    writes[3].pImageInfo = &imgInfo;

    vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  }
}

void ComputePass::updateStorageImage(VkDevice device,
                                     VkImageView storageImageView) {
  VkDescriptorImageInfo imgInfo{};
  imgInfo.imageView = storageImageView;
  imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  for (int f = 0; f < kMaxFramesInFlight; ++f) {
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descriptorSets_[f];
    w.dstBinding = 3;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.descriptorCount = 1;
    w.pImageInfo = &imgInfo;
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
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptorSets_[f];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &nodeBI;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSets_[f];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &matBI;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

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
