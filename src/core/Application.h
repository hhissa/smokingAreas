#pragma once

#include <array>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "CommandPool.h"
#include "ComputePass.h"
#include "StorageImage.h"
#include "Swapchain.h"
#include "SyncObjects.h"
#include "VulkanContext.h"

#include "../gpu/gpuTypes.h"
#include "../gpu/sceneFlattener.h"
#include "../scene/material.h"
#include "../scene/nodeBuilder.h"
#include "../scene/sceneGraph.h"
#include "../scene/sceneValidator.h"

// ─────────────────────────────────────────────────────────────────────────────
// Application
//
// Frame pipeline:
//   [CPU] orbit camera → build SceneUBO → memcpy into UBO slot[frame]
//   [CPU] if scene dirty → flatten → memcpy nodes+materials into SSBOs
//   [CMD] barrier storage image → GENERAL
//   [CMD] dispatch raymarch.comp
//   [CMD] barrier storage image → TRANSFER_SRC
//   [CMD] barrier swapchain image → TRANSFER_DST
//   [CMD] blit
//   [CMD] barrier swapchain image → PRESENT_SRC
//   [CPU] submit → present
// ─────────────────────────────────────────────────────────────────────────────

class Application {
public:
  static constexpr uint32_t kWidth = 1920;
  static constexpr uint32_t kHeight = 1080;
  static constexpr int kMaxFramesInFlight = 2;

  void run();

private:
  // ── GLFW ──────────────────────────────────────────────────────────────────
  GLFWwindow *window_ = nullptr;
  bool framebufferResized_ = false;

  // ── Core ──────────────────────────────────────────────────────────────────
  VulkanContext ctx_;
  Swapchain swapchain_;
  CommandPool commandPool_;
  SyncObjects sync_;
  StorageImage storageImage_;
  ComputePass computePass_;

  // ── Scene ─────────────────────────────────────────────────────────────────
  scene::MaterialRegistry materials_;
  scene::SceneGraph sceneGraph_;
  gpu::SceneFlattener flattener_;
  bool sceneDirty_ = true;
  int32_t bvhRoot_ = -1;

  // ── GPU buffers (HOST_VISIBLE, persistently mapped) ───────────────────────
  // Nodes + materials live in two SSBOs.
  // UBO is double-buffered (one slot per frame-in-flight).
  VkBuffer nodeBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory nodeMemory_ = VK_NULL_HANDLE;
  void *nodeMapped_ = nullptr;
  VkDeviceSize nodeCapacity_ = 0;

  VkBuffer matBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory matMemory_ = VK_NULL_HANDLE;
  void *matMapped_ = nullptr;
  VkDeviceSize matCapacity_ = 0;

  std::array<VkBuffer, kMaxFramesInFlight> uboBuffers_ = {};
  std::array<VkDeviceMemory, kMaxFramesInFlight> uboMemories_ = {};
  std::array<void *, kMaxFramesInFlight> uboMapped_ = {};

  // ── Camera orbit state ────────────────────────────────────────────────────
  float camTheta_ = 0.4f; // horizontal angle (radians)
  float camPhi_ = 0.5f;   // vertical angle   (radians)
  float camRadius_ = 6.0f;

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  void initWindow();
  void initVulkan();
  void initScene();
  void mainLoop();
  void cleanup();

  // ── Frame ─────────────────────────────────────────────────────────────────
  void drawFrame();
  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                           int frameIndex);
  void uploadSceneIfDirty();
  void uploadUBO(int frameIndex, float time);

  // ── Buffer helpers ────────────────────────────────────────────────────────
  void ensureBuffer(VkBuffer &buf, VkDeviceMemory &mem, void *&mapped,
                    VkDeviceSize &capacity, VkDeviceSize needed,
                    VkBufferUsageFlags usage);
  void destroyBuffer(VkBuffer &buf, VkDeviceMemory &mem, void *&mapped);
  VkBuffer createAndMap(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkDeviceMemory &outMem, void *&outMapped);

  // ── Resize ────────────────────────────────────────────────────────────────
  void handleResize();

  static void framebufferResizeCallback(GLFWwindow *w, int, int);
};
;
