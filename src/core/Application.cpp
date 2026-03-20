#include "Application.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

void Application::run() {
  initWindow();
  initVulkan();
  initSystems();
  mainLoop();
  cleanup();
}

void Application::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(kWidth, kHeight, "SDF Renderer", nullptr, nullptr);
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
}

void Application::framebufferResizeCallback(GLFWwindow *w, int, int) {
  reinterpret_cast<Application *>(glfwGetWindowUserPointer(w))
      ->framebufferResized_ = true;
}

void Application::initVulkan() {
  ctx_.init(window_);
  swapchain_.create(ctx_, window_);
  commandPool_.init(ctx_.getDevice(),
                    ctx_.getQueueFamilyIndices().graphicsFamily.value());
  commandPool_.allocateCommandBuffers(ctx_.getDevice(), kMaxFramesInFlight);
  storageImage_.create(ctx_, commandPool_.getPool(), swapchain_.getExtent());

  for (int i = 0; i < kMaxFramesInFlight; ++i)
    uboBuffers_[i] =
        createAndMap(sizeof(gpu::SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     uboMemories_[i], uboMapped_[i]);

  VkDeviceSize nodeInit = sizeof(gpu::GPUNode);
  VkDeviceSize matInit = sizeof(gpu::GPUMaterial);
  nodeBuffer_ = createAndMap(nodeInit, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             nodeMemory_, nodeMapped_);
  nodeCapacity_ = nodeInit;
  matBuffer_ = createAndMap(matInit, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            matMemory_, matMapped_);
  matCapacity_ = matInit;

  computePass_.init(ctx_.getDevice());
  computePass_.writeDescriptors(ctx_.getDevice(), nodeBuffer_, nodeCapacity_,
                                matBuffer_, matCapacity_, uboBuffers_,
                                storageImage_.getView());

  sync_.init(ctx_.getDevice(), kMaxFramesInFlight);
}

void Application::initSystems() {
  cameraSystem_.add("spheres-front", {0.0f, 1.5f, 8.0f}, {0.0f, 0.5f, 0.0f});
  cameraSystem_.add("spheres-above", {0.0f, 8.0f, 4.0f}, {0.0f, 0.0f, 0.0f},
                    65.0f);
  cameraSystem_.add("arches-entry", {0.0f, 1.5f, 9.0f}, {0.0f, 2.0f, 0.0f},
                    74.0f);
  cameraSystem_.add("arches-side", {8.0f, 3.0f, 0.0f}, {0.0f, 2.0f, 0.0f});

  sceneSystem_.add("Spheres", buildSceneSpheres);
  sceneSystem_.add("Arches", buildSceneArches);

  // Load and upload the first scene while the GPU is still idle — before
  // the render loop starts — so the placeholder 1-node buffers are replaced
  // before any frame is submitted.
  sceneSystem_.load(sceneGraph_, materials_);
  sceneDirty_ = true;
  uploadSceneIfDirty();

  spvMtime_ = spvModTime();
}

void Application::mainLoop() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    handleInput();
    checkShaderReload();
    drawFrame();
  }
  vkDeviceWaitIdle(ctx_.getDevice());
}

void Application::handleInput() {
  bool leftNow = glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS;
  bool rightNow = glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS;
  bool upNow = glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS;
  bool downNow = glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS;

  if (leftNow && !leftWas_)
    cameraSystem_.prev();
  if (rightNow && !rightWas_)
    cameraSystem_.next();

  if (upNow && !upWas_) {
    sceneSystem_.prev();
    // Wait for all in-flight work before switching — the scene switch will
    // likely reallocate the node/material buffers.
    vkDeviceWaitIdle(ctx_.getDevice());
    sceneSystem_.load(sceneGraph_, materials_);
    sceneDirty_ = true;
  }
  if (downNow && !downWas_) {
    sceneSystem_.next();
    vkDeviceWaitIdle(ctx_.getDevice());
    sceneSystem_.load(sceneGraph_, materials_);
    sceneDirty_ = true;
  }

  leftWas_ = leftNow;
  rightWas_ = rightNow;
  upWas_ = upNow;
  downWas_ = downNow;
}

void Application::uploadSceneIfDirty() {
  if (!sceneDirty_)
    return;
  sceneDirty_ = false;

  flattener_.flatten(sceneGraph_, materials_);
  bvhRoot_ = flattener_.bvhRoot();

  const auto &nodes = flattener_.nodes();
  const auto &mats = flattener_.gpuMaterials();

  VkDeviceSize nodeBytes = nodes.size() * sizeof(gpu::GPUNode);
  VkDeviceSize matBytes = mats.size() * sizeof(gpu::GPUMaterial);

  bool nodeRebuild = false, matRebuild = false;

  if (nodeBytes > nodeCapacity_) {
    // Both frame slots may be reading this buffer — must be idle before
    // destroy.
    vkDeviceWaitIdle(ctx_.getDevice());
    destroyBuffer(nodeBuffer_, nodeMemory_, nodeMapped_);
    VkDeviceSize newCap =
        std::max(nodeBytes, nodeCapacity_ + nodeCapacity_ / 2);
    nodeBuffer_ = createAndMap(newCap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               nodeMemory_, nodeMapped_);
    nodeCapacity_ = newCap;
    nodeRebuild = true;
  }
  if (matBytes > matCapacity_) {
    if (!nodeRebuild)
      vkDeviceWaitIdle(ctx_.getDevice());
    destroyBuffer(matBuffer_, matMemory_, matMapped_);
    VkDeviceSize newCap = std::max(matBytes, matCapacity_ + matCapacity_ / 2);
    matBuffer_ = createAndMap(newCap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              matMemory_, matMapped_);
    matCapacity_ = newCap;
    matRebuild = true;
  }

  std::memcpy(nodeMapped_, nodes.data(), nodeBytes);
  std::memcpy(matMapped_, mats.data(), matBytes);

  if (nodeRebuild || matRebuild)
    computePass_.rebindBuffers(ctx_.getDevice(), nodeBuffer_, nodeCapacity_,
                               matBuffer_, matCapacity_);
}

void Application::uploadUBO(int frameIndex) {
  VkExtent2D ext = swapchain_.getExtent();
  float aspect = static_cast<float>(ext.width) / static_cast<float>(ext.height);

  gpu::SceneUBO ubo{};
  ubo.view = cameraSystem_.viewMatrix();
  ubo.projection = cameraSystem_.projectionMatrix(aspect);
  ubo.invView = glm::inverse(ubo.view);
  ubo.invProjection = glm::inverse(ubo.projection);
  ubo.cameraPos = glm::vec4(cameraSystem_.position(), 1.0f);
  ubo.resolution = {static_cast<float>(ext.width),
                    static_cast<float>(ext.height), 0.0f, 0.0f};
  ubo.time = static_cast<float>(glfwGetTime());

  std::memcpy(uboMapped_[frameIndex], &ubo, sizeof(ubo));
}

void Application::drawFrame() {
  VkDevice device = ctx_.getDevice();

  VkFence fence = sync_.inFlightFence();
  VkSemaphore imageAvailSem = sync_.imageAvailableSemaphore();

  vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

  uint32_t imageIndex;
  VkResult result =
      vkAcquireNextImageKHR(device, swapchain_.getSwapchain(), UINT64_MAX,
                            imageAvailSem, VK_NULL_HANDLE, &imageIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    handleResize();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    throw std::runtime_error("Failed to acquire swapchain image.");

  vkResetFences(device, 1, &fence);

  int frameIndex = sync_.currentFrame();
  uploadSceneIfDirty();
  uploadUBO(frameIndex);

  VkCommandBuffer cmd = commandPool_.getBuffer(frameIndex);
  vkResetCommandBuffer(cmd, 0);
  recordCommandBuffer(cmd, imageIndex, frameIndex);

  VkSemaphore renderDoneSem = sync_.renderFinishedSemaphore();
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &imageAvailSem;
  si.pWaitDstStageMask = waitStages;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &renderDoneSem;

  if (vkQueueSubmit(ctx_.getGraphicsQueue(), 1, &si, fence) != VK_SUCCESS)
    throw std::runtime_error("Failed to submit command buffer.");

  VkSwapchainKHR swapChains[] = {swapchain_.getSwapchain()};
  VkPresentInfoKHR pi{};
  pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &renderDoneSem;
  pi.swapchainCount = 1;
  pi.pSwapchains = swapChains;
  pi.pImageIndices = &imageIndex;

  result = vkQueuePresentKHR(ctx_.getPresentQueue(), &pi);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebufferResized_) {
    framebufferResized_ = false;
    handleResize();
  } else if (result != VK_SUCCESS)
    throw std::runtime_error("Failed to present.");

  sync_.nextFrame();
}

void Application::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                                      int frameIndex) {
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
    throw std::runtime_error("Failed to begin command buffer.");

  VkExtent2D extent = swapchain_.getExtent();
  VkImage swapImage = swapchain_.getImages()[imageIndex];
  VkImage storeImg = storageImage_.getImage();

  auto imgBarrier = [&](VkImage image, VkImageLayout oldL, VkImageLayout newL,
                        VkAccessFlags srcA, VkAccessFlags dstA,
                        VkPipelineStageFlags s, VkPipelineStageFlags d) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = oldL;
    b.newLayout = newL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = srcA;
    b.dstAccessMask = dstA;
    vkCmdPipelineBarrier(cmd, s, d, 0, 0, nullptr, 0, nullptr, 1, &b);
  };

  imgBarrier(storeImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
             VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  computePass_.dispatch(cmd, frameIndex, extent, bvhRoot_);

  imgBarrier(storeImg, VK_IMAGE_LAYOUT_GENERAL,
             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
             VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
             VK_PIPELINE_STAGE_TRANSFER_BIT);

  imgBarrier(swapImage, VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
             VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
             VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkImageBlit blitRegion{};
  blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blitRegion.srcOffsets[0] = {0, 0, 0};
  blitRegion.srcOffsets[1] = {(int32_t)extent.width, (int32_t)extent.height, 1};
  blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blitRegion.dstOffsets[0] = {0, 0, 0};
  blitRegion.dstOffsets[1] = {(int32_t)extent.width, (int32_t)extent.height, 1};

  vkCmdBlitImage(cmd, storeImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImage,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion,
                 VK_FILTER_NEAREST);

  imgBarrier(swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
             VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    throw std::runtime_error("Failed to end command buffer.");
}

long long Application::spvModTime() {
  struct stat st{};
  if (stat("shaders/raymarch.comp.spv", &st) != 0)
    return 0;
  return static_cast<long long>(st.st_mtime);
}

void Application::checkShaderReload() {
  bool rPressed = (glfwGetKey(window_, GLFW_KEY_R) == GLFW_PRESS);
  bool rTriggered = rPressed && !rKeyWasPressed_;
  rKeyWasPressed_ = rPressed;

  long long mtime = spvModTime();
  bool fileChanged = (mtime != 0 && mtime != spvMtime_);
  if (!rTriggered && !fileChanged)
    return;

  spvMtime_ = mtime;
  vkDeviceWaitIdle(ctx_.getDevice());
  computePass_.reloadPipeline(ctx_.getDevice());
}

void Application::handleResize() {
  swapchain_.recreate(ctx_, window_);
  storageImage_.recreate(ctx_, commandPool_.getPool(), swapchain_.getExtent());
  computePass_.updateStorageImage(ctx_.getDevice(), storageImage_.getView());
}

VkBuffer Application::createAndMap(VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkDeviceMemory &outMem, void *&outMapped) {
  VkDevice device = ctx_.getDevice();
  VkBuffer buf;

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device, &bci, nullptr, &buf) != VK_SUCCESS)
    throw std::runtime_error("Application: failed to create buffer.");

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device, buf, &req);

  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = ctx_.findMemoryType(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (vkAllocateMemory(device, &ai, nullptr, &outMem) != VK_SUCCESS)
    throw std::runtime_error("Application: failed to allocate buffer memory.");

  vkBindBufferMemory(device, buf, outMem, 0);
  vkMapMemory(device, outMem, 0, size, 0, &outMapped);
  return buf;
}

void Application::destroyBuffer(VkBuffer &buf, VkDeviceMemory &mem,
                                void *&mapped) {
  VkDevice device = ctx_.getDevice();
  if (mapped) {
    vkUnmapMemory(device, mem);
    mapped = nullptr;
  }
  if (buf != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, buf, nullptr);
    buf = VK_NULL_HANDLE;
  }
  if (mem != VK_NULL_HANDLE) {
    vkFreeMemory(device, mem, nullptr);
    mem = VK_NULL_HANDLE;
  }
}

void Application::cleanup() {
  VkDevice device = ctx_.getDevice();
  sync_.cleanup(device);
  computePass_.cleanup(device);
  storageImage_.destroy(device);
  destroyBuffer(nodeBuffer_, nodeMemory_, nodeMapped_);
  destroyBuffer(matBuffer_, matMemory_, matMapped_);
  for (int i = 0; i < kMaxFramesInFlight; ++i)
    destroyBuffer(uboBuffers_[i], uboMemories_[i], uboMapped_[i]);
  commandPool_.cleanup(device);
  swapchain_.cleanup(device);
  ctx_.cleanup();
  glfwDestroyWindow(window_);
  glfwTerminate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene builders
// ─────────────────────────────────────────────────────────────────────────────

static scene::CSGNode::Ptr
smoothUnionAll(std::vector<scene::CSGNode::Ptr> nodes, float k) {
  if (nodes.empty())
    return nullptr;
  auto result = std::move(nodes[0]);
  for (size_t i = 1; i < nodes.size(); ++i)
    result = scene::CSGBuilder::smoothU(result, std::move(nodes[i]), k);
  return result;
}

void Application::buildSceneSpheres(scene::SceneGraph &graph,
                                    scene::MaterialRegistry &mats) {
  auto red = mats.get(
      mats.add("red", scene::Material::diffuse({0.9f, 0.15f, 0.1f}, 0.6f)));
  auto gold = mats.get(
      mats.add("gold", scene::Material::metal({1.0f, 0.76f, 0.33f}, 0.2f)));
  auto blue = mats.get(
      mats.add("blue", scene::Material::metal({0.2f, 0.4f, 0.9f}, 0.15f)));
  auto white = mats.get(
      mats.add("white", scene::Material::diffuse({0.9f, 0.9f, 0.9f}, 0.9f)));
  auto glow = mats.get(
      mats.add("glow", scene::Material::emitter({0.8f, 0.5f, 0.1f}, 4.0f)));

  auto s0 = scene::CSGBuilder::sphere(
      "", scene::Transform::translate({0.f, 0.4f, 0.f}), 0.80f, red);
  auto s1 = scene::CSGBuilder::sphere(
      "", scene::Transform::translate({1.1f, 0.f, 0.3f}), 0.50f, gold);
  auto s2 = scene::CSGBuilder::sphere(
      "", scene::Transform::translate({-.8f, 0.2f, 0.5f}), 0.45f, blue);
  auto s3 = scene::CSGBuilder::sphere(
      "", scene::Transform::translate({0.3f, 0.f, -1.0f}), 0.40f, gold);
  auto blob = scene::CSGBuilder::smoothU(
      scene::CSGBuilder::smoothU(scene::CSGBuilder::smoothU(s0, s1, 0.4f), s2,
                                 0.3f),
      s3, 0.25f);

  auto ring = scene::CSGBuilder::torus(
      "", scene::Transform::fromAxisAngle({1, 0, 0}, 75.f), 1.3f, 0.07f, gold);
  auto orb = scene::CSGBuilder::sphere(
      "", scene::Transform::translate({2.5f, 2.0f, -1.5f}), 0.18f, glow);
  auto ground = scene::CSGBuilder::plane("", scene::Transform::identity(),
                                         {0.f, 1.f, 0.f}, 1.0f, white);

  auto mainNode = scene::SceneNode::make("spheres");
  mainNode->csgRoot = scene::CSGBuilder::smoothU(blob, ring, 0.12f);
  mainNode->csgRoot->updateAABB();

  auto orbNode = scene::SceneNode::make("orb");
  orbNode->csgRoot = orb;

  auto groundNode = scene::SceneNode::make("ground");
  groundNode->csgRoot = ground;

  graph.addNode(mainNode);
  graph.addNode(orbNode);
  graph.addNode(groundNode);
}

void Application::buildSceneArches(scene::SceneGraph &graph,
                                   scene::MaterialRegistry &mats) {
  auto stone = mats.get(mats.add(
      "stone", scene::Material::diffuse({0.55f, 0.50f, 0.45f}, 0.88f)));
  auto darkSt = mats.get(mats.add(
      "darkSt", scene::Material::diffuse({0.30f, 0.28f, 0.25f}, 0.92f)));
  auto ground = mats.get(mats.add(
      "ground", scene::Material::diffuse({0.40f, 0.35f, 0.28f}, 0.95f)));

  auto makeArch = [&](float z, float width, float height, float pillarR,
                      const scene::Material &mat) -> scene::CSGNode::Ptr {
    auto pillarL = scene::CSGBuilder::cappedCylinder(
        "", scene::Transform::translate({-width * 0.5f, 0.f, z}), pillarR,
        height, mat);
    auto pillarR_ = scene::CSGBuilder::cappedCylinder(
        "", scene::Transform::translate({width * 0.5f, 0.f, z}), pillarR,
        height, mat);

    scene::Transform lintelT;
    lintelT.position = {0.f, height, z};
    lintelT.rotation =
        glm::angleAxis(glm::half_pi<float>(), glm::vec3(1, 0, 0));
    auto lintel = scene::CSGBuilder::torus("", lintelT, width * 0.5f - pillarR,
                                           pillarR * 1.1f, mat);

    return scene::CSGBuilder::smoothU(
        scene::CSGBuilder::smoothU(pillarL, pillarR_, 0.08f), lintel, 0.12f);
  };

  auto arch0 = makeArch(0.0f, 2.4f, 2.8f, 0.22f, stone);
  auto arch1 = makeArch(-3.5f, 2.0f, 2.5f, 0.19f, darkSt);
  auto arch2 = makeArch(-6.5f, 1.7f, 2.2f, 0.16f, darkSt);

  auto groundPlane = scene::CSGBuilder::plane("", scene::Transform::identity(),
                                              {0.f, 1.f, 0.f}, 0.0f, ground);

  auto archesNode = scene::SceneNode::make("arches");
  archesNode->csgRoot = scene::CSGBuilder::smoothU(
      scene::CSGBuilder::smoothU(arch0, arch1, 0.05f), arch2, 0.05f);
  archesNode->csgRoot->updateAABB();

  auto groundNode = scene::SceneNode::make("ground");
  groundNode->csgRoot = groundPlane;

  graph.addNode(archesNode);
  graph.addNode(groundNode);
}
