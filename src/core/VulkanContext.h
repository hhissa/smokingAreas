#pragma once

#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// ─────────────────────────────────────────────────────────────────────────────
// Shared utility structs used by VulkanContext and Swapchain.
// Placed here because both depend on queue family knowledge.
// ─────────────────────────────────────────────────────────────────────────────

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

// ─────────────────────────────────────────────────────────────────────────────
// VulkanContext
//
// Owns exactly: instance · debug messenger · surface · physical device ·
//               logical device · graphics queue · present queue.
//
// Everything else (swapchain, pipelines, command pools) lives elsewhere.
// Call init() once, then cleanup() at shutdown.
// ─────────────────────────────────────────────────────────────────────────────

class VulkanContext {
public:
    // Initialise the full device stack for the given window.
    void init(GLFWwindow* window);
    void cleanup();

    // ── Accessors ────────────────────────────────────────────────────────────
    VkInstance       getInstance()       const { return instance_; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }
    VkDevice         getDevice()         const { return device_; }
    VkSurfaceKHR     getSurface()        const { return surface_; }
    VkQueue          getGraphicsQueue()  const { return graphicsQueue_; }
    VkQueue          getPresentQueue()   const { return presentQueue_; }

    // Re-queryable helpers used by Swapchain on recreation.
    QueueFamilyIndices    findQueueFamilies(VkPhysicalDevice device) const;
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const;

    // Convenience: the indices for the currently selected physical device.
    QueueFamilyIndices getQueueFamilyIndices() const {
        return findQueueFamilies(physicalDevice_);
    }

    // Find a memory type index satisfying the given filter and property flags.
    // Used by StorageImage and any future buffer/image allocators.
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

    // Submit a one-shot command buffer synchronously on the graphics queue.
    // Usage:
    //   VkCommandBuffer cmd = ctx.beginSingleTimeCommands(pool);
    //   // record commands...
    //   ctx.endSingleTimeCommands(pool, cmd);
    VkCommandBuffer beginSingleTimeCommands(VkCommandPool pool) const;
    void            endSingleTimeCommands(VkCommandPool pool, VkCommandBuffer cmd) const;

private:
    // ── Core handles ─────────────────────────────────────────────────────────
    VkInstance               instance_        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_  = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_         = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice_  = VK_NULL_HANDLE;
    VkDevice                 device_          = VK_NULL_HANDLE;
    VkQueue                  graphicsQueue_   = VK_NULL_HANDLE;
    VkQueue                  presentQueue_    = VK_NULL_HANDLE;

    // ── Init helpers ─────────────────────────────────────────────────────────
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createLogicalDevice();

    // ── Physical device evaluation ───────────────────────────────────────────
    bool isDeviceSuitable(VkPhysicalDevice device) const;
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) const;

    // ── Instance helpers ─────────────────────────────────────────────────────
    std::vector<const char*> getRequiredExtensions() const;
    bool checkValidationLayerSupport() const;
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& ci) const;

    // ── Debug messenger proxy fns ────────────────────────────────────────────
    static VkResult createDebugUtilsMessengerEXT(
        VkInstance instance,
        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDebugUtilsMessengerEXT* pDebugMessenger);

    static void destroyDebugUtilsMessengerEXT(
        VkInstance instance,
        VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};
