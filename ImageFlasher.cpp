#include "ImageFlasher.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Inline SPIR-V
// Compile your own with:
//   glslc image_flash.vert -o vert_flash.spv
//   glslc image_flash.frag -o frag_flash.spv
// Then xxd -i vert_flash.spv  to get the byte array.
//
// For convenience, load from files at runtime here:
// ---------------------------------------------------------------------------

static std::vector<char> readSPV(const std::string &path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("ImageFlasher: failed to open shader: " + path);
  size_t sz = (size_t)file.tellg();
  std::vector<char> buf(sz);
  file.seekg(0);
  file.read(buf.data(), sz);
  return buf;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void ImageFlasher::init(VkDevice device_, VkPhysicalDevice physicalDevice_,
                        VkCommandPool commandPool_, VkQueue graphicsQueue_,
                        VkRenderPass renderPass, VkExtent2D swapChainExtent,
                        const std::vector<std::string> &imagePaths) {
  device = device_;
  physicalDevice = physicalDevice_;
  commandPool = commandPool_;
  graphicsQueue = graphicsQueue_;

  images.resize(imagePaths.size());
  for (size_t i = 0; i < imagePaths.size(); i++)
    loadImage(imagePaths[i], images[i]);

  createDescriptorSetLayout();
  createDescriptorPool((int)images.size());
  allocateDescriptorSets();
  createPipeline(renderPass, swapChainExtent);
}

void ImageFlasher::draw(VkCommandBuffer commandBuffer, int flashIndex) {
  if (images.empty())
    return;
  flashIndex = flashIndex % (int)images.size();

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 1,
                          &images[flashIndex].descriptorSet, 0, nullptr);
  // Fullscreen triangle — no vertex buffer needed
  vkCmdDraw(commandBuffer, 3, 1, 0, 0);
}

void ImageFlasher::onSwapchainRecreate(VkRenderPass renderPass,
                                       VkExtent2D swapChainExtent) {
  destroyPipeline();
  createPipeline(renderPass, swapChainExtent);
}

void ImageFlasher::cleanup() {
  destroyPipeline();

  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

  for (auto &img : images) {
    vkDestroySampler(device, img.sampler, nullptr);
    vkDestroyImageView(device, img.view, nullptr);
    vkDestroyImage(device, img.image, nullptr);
    vkFreeMemory(device, img.memory, nullptr);
  }
  images.clear();
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

void ImageFlasher::loadImage(const std::string &path, ImageData &out) {
  int w, h, channels;
  stbi_uc *pixels = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
  if (!pixels)
    throw std::runtime_error("ImageFlasher: failed to load image: " + path);

  VkDeviceSize imageSize = w * h * 4;

  // Staging buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;
  createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               stagingBuffer, stagingMemory);

  void *data;
  vkMapMemory(device, stagingMemory, 0, imageSize, 0, &data);
  memcpy(data, pixels, (size_t)imageSize);
  vkUnmapMemory(device, stagingMemory);
  stbi_image_free(pixels);

  // Create VkImage
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent = {(uint32_t)w, (uint32_t)h, 1};
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

  if (vkCreateImage(device, &imageInfo, nullptr, &out.image) != VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to create image");

  VkMemoryRequirements memReq;
  vkGetImageMemoryRequirements(device, out.image, &memReq);
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReq.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (vkAllocateMemory(device, &allocInfo, nullptr, &out.memory) != VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to allocate image memory");
  vkBindImageMemory(device, out.image, out.memory, 0);

  transitionImageLayout(out.image, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  copyBufferToImage(stagingBuffer, out.image, (uint32_t)w, (uint32_t)h);
  transitionImageLayout(out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingMemory, nullptr);

  // Image view
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = out.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;
  if (vkCreateImageView(device, &viewInfo, nullptr, &out.view) != VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to create image view");

  // Sampler
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  if (vkCreateSampler(device, &samplerInfo, nullptr, &out.sampler) !=
      VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to create sampler");
}

// ---------------------------------------------------------------------------
// Descriptor layout / pool / sets
// ---------------------------------------------------------------------------

void ImageFlasher::createDescriptorSetLayout() {
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  info.bindingCount = 1;
  info.pBindings = &binding;

  if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                  &descriptorSetLayout) != VK_SUCCESS)
    throw std::runtime_error(
        "ImageFlasher: failed to create descriptor set layout");
}

void ImageFlasher::createDescriptorPool(int imageCount) {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSize.descriptorCount = (uint32_t)imageCount;

  VkDescriptorPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  info.poolSizeCount = 1;
  info.pPoolSizes = &poolSize;
  info.maxSets = (uint32_t)imageCount;

  if (vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool) !=
      VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to create descriptor pool");
}

void ImageFlasher::allocateDescriptorSets() {
  std::vector<VkDescriptorSetLayout> layouts(images.size(),
                                             descriptorSetLayout);

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = (uint32_t)images.size();
  allocInfo.pSetLayouts = layouts.data();

  std::vector<VkDescriptorSet> sets(images.size());
  if (vkAllocateDescriptorSets(device, &allocInfo, sets.data()) != VK_SUCCESS)
    throw std::runtime_error(
        "ImageFlasher: failed to allocate descriptor sets");

  for (size_t i = 0; i < images.size(); i++) {
    images[i].descriptorSet = sets[i];

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = images[i].view;
    imageInfo.sampler = images[i].sampler;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = images[i].descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }
}

// ---------------------------------------------------------------------------
// Pipeline — loads image_flash.vert.spv / image_flash.frag.spv
// ---------------------------------------------------------------------------

void ImageFlasher::createPipeline(VkRenderPass renderPass, VkExtent2D extent) {
  auto vertCode = readSPV("image_flash.vert.spv");
  auto fragCode = readSPV("image_flash.frag.spv");

  auto makeModule = [&](const std::vector<char> &code) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
      throw std::runtime_error("ImageFlasher: failed to create shader module");
    return mod;
  };

  VkShaderModule vertModule = makeModule(vertCode);
  VkShaderModule fragModule = makeModule(fragCode);

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vertModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fragModule;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertexInput{};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkViewport viewport{0, 0, (float)extent.width, (float)extent.height, 0, 1};
  VkRect2D scissor{{0, 0}, extent};

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blendAttachment{};
  blendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &blendAttachment;

  // Dynamic viewport/scissor so swapchain recreate is easy
  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = (uint32_t)dynamicStates.size();
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &descriptorSetLayout;

  if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) !=
      VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to create pipeline layout");

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = stages;
  pipelineInfo.pVertexInputState = &vertexInput;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = renderPass;
  pipelineInfo.subpass = 0;

  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                nullptr, &pipeline) != VK_SUCCESS)
    throw std::runtime_error(
        "ImageFlasher: failed to create graphics pipeline");

  vkDestroyShaderModule(device, vertModule, nullptr);
  vkDestroyShaderModule(device, fragModule, nullptr);
}

void ImageFlasher::destroyPipeline() {
  if (pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, pipeline, nullptr);
    pipeline = VK_NULL_HANDLE;
  }
  if (pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

uint32_t ImageFlasher::findMemoryType(uint32_t typeFilter,
                                      VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    if ((typeFilter & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  throw std::runtime_error("ImageFlasher: failed to find suitable memory type");
}

void ImageFlasher::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags properties,
                                VkBuffer &buffer,
                                VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to create buffer");

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(device, buffer, &memReq);
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReq.size;
  allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties);
  if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS)
    throw std::runtime_error("ImageFlasher: failed to allocate buffer memory");
  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

VkCommandBuffer ImageFlasher::beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer cb;
  vkAllocateCommandBuffers(device, &allocInfo, &cb);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &beginInfo);
  return cb;
}

void ImageFlasher::endSingleTimeCommands(VkCommandBuffer cb) {
  vkEndCommandBuffer(cb);
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cb;
  vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);
  vkFreeCommandBuffers(device, commandPool, 1, &cb);
}

void ImageFlasher::transitionImageLayout(VkImage image, VkImageLayout oldLayout,
                                         VkImageLayout newLayout) {
  VkCommandBuffer cb = beginSingleTimeCommands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags srcStage, dstStage;
  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else {
    throw std::runtime_error("ImageFlasher: unsupported layout transition");
  }

  vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);
  endSingleTimeCommands(cb);
}

void ImageFlasher::copyBufferToImage(VkBuffer buffer, VkImage image,
                                     uint32_t width, uint32_t height) {
  VkCommandBuffer cb = beginSingleTimeCommands();

  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(cb, buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  endSingleTimeCommands(cb);
}
