#pragma once

#include <array>
#include <sys/stat.h>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "CameraSystem.h"
#include "CommandPool.h"
#include "ComputePass.h"
#include "SceneSystem.h"
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
// Controls:
//   Left / Right arrows  — previous / next camera
//   Down / Up arrows     — previous / next scene
//   R                    — force shader reload
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

  // ── Scene and camera systems ──────────────────────────────────────────────
  scene::MaterialRegistry materials_;
  scene::SceneGraph sceneGraph_;
  gpu::SceneFlattener flattener_;
  bool sceneDirty_ = true;
  int32_t bvhRoot_ = -1;

  CameraSystem cameraSystem_;
  SceneSystem sceneSystem_;

  // Key edge-detection state
  bool leftWas_ = false;
  bool rightWas_ = false;
  bool upWas_ = false;
  bool downWas_ = false;

  // ── GPU buffers ───────────────────────────────────────────────────────────
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

  // ── Shader hot-reload ─────────────────────────────────────────────────────
  long long spvMtime_ = 0;
  bool rKeyWasPressed_ = false;

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  void initWindow();
  void initVulkan();
  void initSystems(); // register all cameras and scenes
  void mainLoop();
  void cleanup();

  // ── Frame ─────────────────────────────────────────────────────────────────
  void drawFrame();
  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                           int frameIndex);
  void uploadSceneIfDirty();
  void uploadUBO(int frameIndex);
  void handleInput();

  // ── Scene builders ─────────────────────────────────────────────────────────
  // Each registered with sceneSystem_.add().
  static void buildSceneTree(scene::SceneGraph &, scene::MaterialRegistry &);
  static void buildSceneSpheres(scene::SceneGraph &, scene::MaterialRegistry &);
  static void buildSceneArches(scene::SceneGraph &, scene::MaterialRegistry &);

  // ── Resize / reload ───────────────────────────────────────────────────────
  void handleResize();
  void checkShaderReload();

  // ── Buffer helpers ────────────────────────────────────────────────────────
  VkBuffer createAndMap(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkDeviceMemory &outMem, void *&outMapped);
  void destroyBuffer(VkBuffer &buf, VkDeviceMemory &mem, void *&mapped);

  static long long spvModTime();
  static void framebufferResizeCallback(GLFWwindow *w, int, int);
};
