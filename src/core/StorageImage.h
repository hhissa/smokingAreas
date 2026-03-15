#pragma once

#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "VulkanContext.h"

// ─────────────────────────────────────────────────────────────────────────────
// StorageImage
//
// A device-local RGBA8 image that the compute shader writes to and the blit
// step reads from.  One instance covers the full swapchain extent.
//
// Lifecycle:
//   create()   — called once after VulkanContext and CommandPool are up
//   destroy()  — called before recreate() or at shutdown
//   recreate() — called whenever the swapchain extent changes
//
// Format is exposed as kFormat so ComputePass can declare its descriptor
// with the correct VkFormat without hard-coding a magic number.
// ─────────────────────────────────────────────────────────────────────────────

class StorageImage {
public:
    // VK_FORMAT_R8G8B8A8_UNORM matches the `rgba8` GLSL image format qualifier.
    static constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

    void create(const VulkanContext& ctx, VkCommandPool cmdPool, VkExtent2D extent);
    void destroy(VkDevice device);

    // Convenience wrapper: destroy then create.
    void recreate(const VulkanContext& ctx, VkCommandPool cmdPool, VkExtent2D extent) {
        destroy(ctx.getDevice());
        create(ctx, cmdPool, extent);
    }

    VkImage     getImage()  const { return image_;  }
    VkImageView getView()   const { return view_;   }
    VkExtent2D  getExtent() const { return extent_; }

private:
    VkImage        image_  = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView    view_   = VK_NULL_HANDLE;
    VkExtent2D     extent_ = {};
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation (header-only — StorageImage is simple enough)
// ─────────────────────────────────────────────────────────────────────────────

inline void StorageImage::create(const VulkanContext& ctx,
                                  VkCommandPool cmdPool,
                                  VkExtent2D extent)
{
    extent_ = extent;
    VkDevice device = ctx.getDevice();

    // ── Create the image ─────────────────────────────────────────────────────
    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = kFormat;
    imageCI.extent        = { extent.width, extent.height, 1 };
    imageCI.mipLevels     = 1;
    imageCI.arrayLayers   = 1;
    imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE_BIT  — compute shader writes via imageStore()
    // TRANSFER_SRC — blit source when copying to the swapchain image
    imageCI.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imageCI, nullptr, &image_) != VK_SUCCESS)
        throw std::runtime_error("StorageImage: failed to create image.");

    // ── Allocate device-local memory ─────────────────────────────────────────
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, image_, &memReqs);

    VkMemoryAllocateInfo allocCI{};
    allocCI.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocCI.allocationSize  = memReqs.size;
    allocCI.memoryTypeIndex = ctx.findMemoryType(memReqs.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocCI, nullptr, &memory_) != VK_SUCCESS)
        throw std::runtime_error("StorageImage: failed to allocate memory.");

    vkBindImageMemory(device, image_, memory_, 0);

    // ── Create the image view ─────────────────────────────────────────────────
    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = image_;
    viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format                          = kFormat;
    viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = 1;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(device, &viewCI, nullptr, &view_) != VK_SUCCESS)
        throw std::runtime_error("StorageImage: failed to create image view.");

    // ── Transition to VK_IMAGE_LAYOUT_GENERAL ────────────────────────────────
    // The compute shader requires GENERAL layout to use imageStore().
    // We perform this once here; from this point the image stays in GENERAL
    // for the lifetime of this StorageImage instance.
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands(cmdPool);

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image_;
    barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask       = 0;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.endSingleTimeCommands(cmdPool, cmd);
}

inline void StorageImage::destroy(VkDevice device) {
    if (view_   != VK_NULL_HANDLE) vkDestroyImageView(device, view_,   nullptr);
    if (image_  != VK_NULL_HANDLE) vkDestroyImage    (device, image_,  nullptr);
    if (memory_ != VK_NULL_HANDLE) vkFreeMemory      (device, memory_, nullptr);
    view_   = VK_NULL_HANDLE;
    image_  = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
}
