#include "VulkanContext.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time validation layer toggle
// ─────────────────────────────────────────────────────────────────────────────
#ifdef NDEBUG
    static constexpr bool kEnableValidationLayers = false;
#else
    static constexpr bool kEnableValidationLayers = true;
#endif

static const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

void VulkanContext::init(GLFWwindow* window) {
    createInstance();
    setupDebugMessenger();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
}

void VulkanContext::cleanup() {
    vkDestroyDevice(device_, nullptr);

    if (kEnableValidationLayers) {
        destroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
    }

    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Instance
// ─────────────────────────────────────────────────────────────────────────────

void VulkanContext::createInstance() {
    if (kEnableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested but not available.");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "SDF Renderer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName        = "No Engine";
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // The debug messenger create-info is chained into pNext so that we get
    // validation coverage for vkCreateInstance / vkDestroyInstance themselves.
    VkDebugUtilsMessengerCreateInfoEXT debugCI{};
    if (kEnableValidationLayers) {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
        populateDebugMessengerCreateInfo(debugCI);
        createInfo.pNext = &debugCI;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext             = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug messenger
// ─────────────────────────────────────────────────────────────────────────────

void VulkanContext::setupDebugMessenger() {
    if (!kEnableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    if (createDebugUtilsMessengerEXT(instance_, &createInfo, nullptr, &debugMessenger_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to set up debug messenger.");
    }
}

void VulkanContext::populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT& ci) const
{
    ci = {};
    ci.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
}

// ─────────────────────────────────────────────────────────────────────────────
// Surface
// ─────────────────────────────────────────────────────────────────────────────

void VulkanContext::createSurface(GLFWwindow* window) {
    if (glfwCreateWindowSurface(instance_, window, nullptr, &surface_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Physical device
// ─────────────────────────────────────────────────────────────────────────────

void VulkanContext::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No GPUs with Vulkan support found.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice_ = device;
            return;
        }
    }

    throw std::runtime_error("No suitable GPU found.");
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice device) const {
    QueueFamilyIndices indices = findQueueFamilies(device);
    if (!indices.isComplete()) return false;
    if (!checkDeviceExtensionSupport(device)) return false;

    SwapChainSupportDetails sc = querySwapChainSupport(device);
    return !sc.formats.empty() && !sc.presentModes.empty();
}

bool VulkanContext::checkDeviceExtensionSupport(VkPhysicalDevice device) const {
    uint32_t count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);

    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());

    std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (const auto& ext : available) {
        required.erase(ext.extensionName);
    }
    return required.empty();
}

// ─────────────────────────────────────────────────────────────────────────────
// Logical device
// ─────────────────────────────────────────────────────────────────────────────

void VulkanContext::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };

    float priority = 1.0f;
    for (uint32_t family : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &priority;
        queueCreateInfos.push_back(qi);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.pEnabledFeatures        = &deviceFeatures;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    if (kEnableValidationLayers) {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device.");
    }

    vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, indices.presentFamily.value(),  0, &presentQueue_);
}

// ─────────────────────────────────────────────────────────────────────────────
// Queue family / swapchain support queries (called by Swapchain too)
// ─────────────────────────────────────────────────────────────────────────────

QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice device) const {
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);

    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) break;
    }
    return indices;
}

SwapChainSupportDetails VulkanContext::querySwapChainSupport(VkPhysicalDevice device) const {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, details.presentModes.data());
    }

    return details;
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory helpers
// ─────────────────────────────────────────────────────────────────────────────

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-time command helpers
// ─────────────────────────────────────────────────────────────────────────────

VkCommandBuffer VulkanContext::beginSingleTimeCommands(VkCommandPool pool) const {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool        = pool;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandPool pool, VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue_);
    vkFreeCommandBuffers(device_, pool, 1, &cmd);
}



std::vector<const char*> VulkanContext::getRequiredExtensions() const {
    uint32_t glfwCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);

    std::vector<const char*> extensions(glfwExts, glfwExts + glfwCount);
    if (kEnableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

bool VulkanContext::checkValidationLayerSupport() const {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> available(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, available.data());

    for (const char* name : kValidationLayers) {
        bool found = false;
        for (const auto& prop : available) {
            if (strcmp(name, prop.layerName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug messenger proxy functions (loaded at runtime via vkGetInstanceProcAddr)
// ─────────────────────────────────────────────────────────────────────────────

VkResult VulkanContext::createDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    return fn ? fn(instance, pCreateInfo, pAllocator, pDebugMessenger)
              : VK_ERROR_EXTENSION_NOT_PRESENT;
}

void VulkanContext::destroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator)
{
    auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (fn) fn(instance, debugMessenger, pAllocator);
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
    VkDebugUtilsMessageTypeFlagsEXT       /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/)
{
    std::cerr << "[Vulkan] " << pCallbackData->pMessage << "\n";
    return VK_FALSE;
}
