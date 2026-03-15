#pragma once

#include <stdexcept>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// ─────────────────────────────────────────────────────────────────────────────
// SyncObjects
//
// Owns per-frame-in-flight:
//   imageAvailableSemaphore  — signalled when a swapchain image is ready
//   renderFinishedSemaphore  — signalled when the command buffer has finished
//   inFlightFence            — CPU gate, ensures we don't overwrite a frame
//                              that the GPU is still processing
//
// Also owns currentFrame_, which advances with nextFrame() so that the
// frame index is always controlled from one place.
// ─────────────────────────────────────────────────────────────────────────────

class SyncObjects {
public:
    void init(VkDevice device, int framesInFlight);
    void cleanup(VkDevice device);

    // Advance to the next frame-in-flight slot (wraps at framesInFlight).
    void nextFrame() {
        currentFrame_ = (currentFrame_ + 1) % framesInFlight_;
    }

    // ── Per-frame accessors ───────────────────────────────────────────────
    uint32_t currentFrame() const { return currentFrame_; }

    VkSemaphore imageAvailableSemaphore() const {
        return imageAvailableSemaphores_[currentFrame_];
    }
    VkSemaphore renderFinishedSemaphore() const {
        return renderFinishedSemaphores_[currentFrame_];
    }
    VkFence inFlightFence() const {
        return inFlightFences_[currentFrame_];
    }

private:
    int                       framesInFlight_ = 2;
    uint32_t                  currentFrame_   = 0;
    std::vector<VkSemaphore>  imageAvailableSemaphores_;
    std::vector<VkSemaphore>  renderFinishedSemaphores_;
    std::vector<VkFence>      inFlightFences_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation (header-only for this small class)
// ─────────────────────────────────────────────────────────────────────────────

inline void SyncObjects::init(VkDevice device, int framesInFlight) {
    framesInFlight_ = framesInFlight;
    imageAvailableSemaphores_.resize(framesInFlight);
    renderFinishedSemaphores_.resize(framesInFlight);
    inFlightFences_.resize(framesInFlight);

    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    // Fences start signalled so the first vkWaitForFences call returns
    // immediately (there is no "previous frame" to wait for on frame 0).
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < framesInFlight; ++i) {
        if (vkCreateSemaphore(device, &si, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS ||
            vkCreateFence    (device, &fi, nullptr, &inFlightFences_[i])            != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create synchronisation objects.");
        }
    }
}

inline void SyncObjects::cleanup(VkDevice device) {
    for (int i = 0; i < framesInFlight_; ++i) {
        vkDestroySemaphore(device, renderFinishedSemaphores_[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores_[i], nullptr);
        vkDestroyFence    (device, inFlightFences_[i],           nullptr);
    }
}
