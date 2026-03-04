#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

class ImageFlasher {
public:
    // Call once after logical device is created
    void init(VkDevice device, VkPhysicalDevice physicalDevice,
              VkCommandPool commandPool, VkQueue graphicsQueue,
              VkRenderPass renderPass, VkExtent2D swapChainExtent,
              const std::vector<std::string>& imagePaths);

    // Call in recordCommandBuffer when state >= 5
    // flashIndex = which image to show (compute from localTime on CPU)
    void draw(VkCommandBuffer commandBuffer, int flashIndex);

    // Call on swapchain recreate
    void onSwapchainRecreate(VkRenderPass renderPass, VkExtent2D swapChainExtent);

    void cleanup();

private:
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;

    // Per-image resources
    struct ImageData {
        VkImage image;
        VkDeviceMemory memory;
        VkImageView view;
        VkSampler sampler;
        VkDescriptorSet descriptorSet;
    };
    std::vector<ImageData> images;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Helpers
    void loadImage(const std::string& path, ImageData& out);
    void createDescriptorSetLayout();
    void createDescriptorPool(int imageCount);
    void allocateDescriptorSets();
    void createPipeline(VkRenderPass renderPass, VkExtent2D extent);
    void destroyPipeline();

    VkShaderModule createShaderModule(const std::vector<uint32_t>& code);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
};
