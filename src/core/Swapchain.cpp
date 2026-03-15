#include "Swapchain.h"

#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

void Swapchain::create(const VulkanContext& ctx, GLFWwindow* window, VkRenderPass renderPass) {
    createSwapchain(ctx, window);
    createImageViews(ctx.getDevice());
    if (renderPass != VK_NULL_HANDLE)
        createFramebuffers(ctx.getDevice(), renderPass);
}

void Swapchain::cleanup(VkDevice device) {
    for (auto fb   : framebuffers_) vkDestroyFramebuffer(device, fb,   nullptr);
    for (auto view : imageViews_)   vkDestroyImageView  (device, view, nullptr);
    vkDestroySwapchainKHR(device, swapChain_, nullptr);

    framebuffers_.clear();
    imageViews_.clear();
    images_.clear();
    swapChain_ = VK_NULL_HANDLE;
}

void Swapchain::recreate(const VulkanContext& ctx, GLFWwindow* window, VkRenderPass renderPass) {
    // Block while the window is minimised (extent would be 0×0).
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(window, &w, &h);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(ctx.getDevice());
    cleanup(ctx.getDevice());
    create(ctx, window, renderPass);
}

// ─────────────────────────────────────────────────────────────────────────────
// Swapchain creation
// ─────────────────────────────────────────────────────────────────────────────

void Swapchain::createSwapchain(const VulkanContext& ctx, GLFWwindow* window) {
    SwapChainSupportDetails support = ctx.querySwapChainSupport(ctx.getPhysicalDevice());

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    VkPresentModeKHR   presentMode   = chooseSwapPresentMode(support.presentModes);
    VkExtent2D         extent        = chooseSwapExtent(support.capabilities, window);

    // Request one more image than the minimum to avoid stalling on the driver.
    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0) {
        imageCount = std::min(imageCount, support.capabilities.maxImageCount);
    }

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = ctx.getSurface();
    ci.minImageCount    = imageCount;
    ci.imageFormat      = surfaceFormat.format;
    ci.imageColorSpace  = surfaceFormat.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT; // blit target from compute output

    QueueFamilyIndices indices = ctx.getQueueFamilyIndices();
    uint32_t queueFamilyIndices[] = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };

    if (indices.graphicsFamily != indices.presentFamily) {
        // Two distinct queue families — share the image between them.
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
        // Same family — exclusive ownership, no synchronisation overhead.
        ci.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
        ci.queueFamilyIndexCount = 0;
        ci.pQueueFamilyIndices   = nullptr;
    }

    ci.preTransform   = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = presentMode;
    ci.clipped        = VK_TRUE;
    ci.oldSwapchain   = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(ctx.getDevice(), &ci, nullptr, &swapChain_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swap chain.");
    }

    // Retrieve the actual image handles allocated by the driver.
    uint32_t count;
    vkGetSwapchainImagesKHR(ctx.getDevice(), swapChain_, &count, nullptr);
    images_.resize(count);
    vkGetSwapchainImagesKHR(ctx.getDevice(), swapChain_, &count, images_.data());

    imageFormat_ = surfaceFormat.format;
    extent_      = extent;
}

// ─────────────────────────────────────────────────────────────────────────────
// Image views
// ─────────────────────────────────────────────────────────────────────────────

void Swapchain::createImageViews(VkDevice device) {
    imageViews_.resize(images_.size());
    for (size_t i = 0; i < images_.size(); ++i) {
        VkImageViewCreateInfo ci{};
        ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image                           = images_[i];
        ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        ci.format                          = imageFormat_;
        ci.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.baseMipLevel   = 0;
        ci.subresourceRange.levelCount     = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(device, &ci, nullptr, &imageViews_[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view.");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Framebuffers
// ─────────────────────────────────────────────────────────────────────────────

void Swapchain::createFramebuffers(VkDevice device, VkRenderPass renderPass) {
    framebuffers_.resize(imageViews_.size());
    for (size_t i = 0; i < imageViews_.size(); ++i) {
        VkImageView attachments[] = { imageViews_[i] };

        VkFramebufferCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass      = renderPass;
        ci.attachmentCount = 1;
        ci.pAttachments    = attachments;
        ci.width           = extent_.width;
        ci.height          = extent_.height;
        ci.layers          = 1;

        if (vkCreateFramebuffer(device, &ci, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer.");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format / mode / extent selection
// ─────────────────────────────────────────────────────────────────────────────

VkSurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& available)
{
    // Prefer sRGB for correct gamma handling in the UI / text pass.
    for (const auto& fmt : available) {
        if (fmt.format     == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return fmt;
        }
    }
    return available[0];
}

VkPresentModeKHR Swapchain::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& available)
{
    // Mailbox: triple-buffer, lowest latency without tearing.
    // Fall back to FIFO (guaranteed present on all drivers).
    for (const auto& mode : available) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Swapchain::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR& capabilities,
    GLFWwindow* window)
{
    // If the driver has set a concrete extent, use it directly.
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    // Otherwise query the framebuffer size in pixels (differs from screen
    // coordinates on high-DPI displays) and clamp to the supported range.
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);

    VkExtent2D actual = {
        static_cast<uint32_t>(w),
        static_cast<uint32_t>(h)
    };

    actual.width  = std::clamp(actual.width,
                               capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width);
    actual.height = std::clamp(actual.height,
                               capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
    return actual;
}
