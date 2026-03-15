#pragma once

#include <stdexcept>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// ─────────────────────────────────────────────────────────────────────────────
// CommandPool
//
// Owns: VkCommandPool and the primary VkCommandBuffers allocated from it.
//
// The pool is created with RESET_COMMAND_BUFFER_BIT so that individual
// buffers can be re-recorded each frame without resetting the whole pool.
// ─────────────────────────────────────────────────────────────────────────────

class CommandPool {
public:
    // Create the pool bound to graphicsQueueFamily.
    void init(VkDevice device, uint32_t graphicsQueueFamily);

    // Allocate `count` primary command buffers from the pool.
    // Safe to call once after init().
    void allocateCommandBuffers(VkDevice device, uint32_t count);

    void cleanup(VkDevice device);

    // ── Accessors ────────────────────────────────────────────────────────────
    VkCommandPool getPool() const { return pool_; }

    // Returns the command buffer for the given frame-in-flight index.
    VkCommandBuffer getBuffer(uint32_t frameIndex) const {
        return commandBuffers_[frameIndex];
    }

    const std::vector<VkCommandBuffer>& getBuffers() const {
        return commandBuffers_;
    }

private:
    VkCommandPool                  pool_           = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer>   commandBuffers_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation (header-only for this small class)
// ─────────────────────────────────────────────────────────────────────────────

inline void CommandPool::init(VkDevice device, uint32_t graphicsQueueFamily) {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphicsQueueFamily;

    if (vkCreateCommandPool(device, &ci, nullptr, &pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool.");
    }
}

inline void CommandPool::allocateCommandBuffers(VkDevice device, uint32_t count) {
    commandBuffers_.resize(count);

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = count;

    if (vkAllocateCommandBuffers(device, &ai, commandBuffers_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers.");
    }
}

inline void CommandPool::cleanup(VkDevice device) {
    // Command buffers are freed implicitly when the pool is destroyed.
    vkDestroyCommandPool(device, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    commandBuffers_.clear();
}
