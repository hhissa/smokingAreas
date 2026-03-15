#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "VulkanContext.h"

// ─────────────────────────────────────────────────────────────────────────────
// Swapchain
//
// Owns: swapchain · swap images (borrowed from driver) · image views ·
//       framebuffers.
//
// Framebuffers are tied to a render pass, so the pass handle must be provided
// at creation time.  On resize, call recreate() with the same render pass.
// ─────────────────────────────────────────────────────────────────────────────

class Swapchain {
public:
    // Create the full swapchain stack.
    // Pass VK_NULL_HANDLE for renderPass to skip framebuffer creation
    // (compute-only path — framebuffers are never needed).
    void create(const VulkanContext& ctx, GLFWwindow* window,
                VkRenderPass renderPass = VK_NULL_HANDLE);

    // Destroy image views and framebuffers, then the swapchain itself.
    void cleanup(VkDevice device);

    // Tear down and rebuild in-place (e.g. after a window resize).
    void recreate(const VulkanContext& ctx, GLFWwindow* window,
                  VkRenderPass renderPass = VK_NULL_HANDLE);

    // ── Accessors ────────────────────────────────────────────────────────────
    VkSwapchainKHR             getSwapchain()   const { return swapChain_; }
    VkFormat                   getImageFormat() const { return imageFormat_; }
    VkExtent2D                 getExtent()      const { return extent_; }
    const std::vector<VkImage>&       getImages()      const { return images_; }
    const std::vector<VkImageView>&   getImageViews()  const { return imageViews_; }
    const std::vector<VkFramebuffer>& getFramebuffers()const { return framebuffers_; }

private:
    VkSwapchainKHR             swapChain_    = VK_NULL_HANDLE;
    VkFormat                   imageFormat_  = VK_FORMAT_UNDEFINED;
    VkExtent2D                 extent_       = {};
    std::vector<VkImage>       images_;
    std::vector<VkImageView>   imageViews_;
    std::vector<VkFramebuffer> framebuffers_;

    // Called by create() / recreate()
    void createSwapchain(const VulkanContext& ctx, GLFWwindow* window);
    void createImageViews(VkDevice device);
    void createFramebuffers(VkDevice device, VkRenderPass renderPass);

    // Format / mode / extent selection
    static VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& available);

    static VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& available);

    static VkExtent2D chooseSwapExtent(
        const VkSurfaceCapabilitiesKHR& capabilities,
        GLFWwindow* window);
};
