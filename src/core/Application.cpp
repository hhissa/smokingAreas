#include "./Application.h"

#include <cstring>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Entry
// ─────────────────────────────────────────────────────────────────────────────

void Application::run() {
    initWindow();
    initVulkan();
    initScene();
    mainLoop();
    cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// Window
// ─────────────────────────────────────────────────────────────────────────────

void Application::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(kWidth, kHeight, "SDF Renderer", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
}

void Application::framebufferResizeCallback(GLFWwindow* w, int, int) {
    reinterpret_cast<Application*>(glfwGetWindowUserPointer(w))->framebufferResized_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan init
// ─────────────────────────────────────────────────────────────────────────────

void Application::initVulkan() {
    ctx_.init(window_);
    swapchain_.create(ctx_, window_);

    commandPool_.init(ctx_.getDevice(),
                      ctx_.getQueueFamilyIndices().graphicsFamily.value());
    commandPool_.allocateCommandBuffers(ctx_.getDevice(), kMaxFramesInFlight);

    storageImage_.create(ctx_, commandPool_.getPool(), swapchain_.getExtent());

    // ── UBO buffers (one per frame slot) ──────────────────────────────────────
    for (int i = 0; i < kMaxFramesInFlight; ++i)
        uboBuffers_[i] = createAndMap(sizeof(gpu::SceneUBO),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       uboMemories_[i], uboMapped_[i]);

    // ── Placeholder SSBO buffers (1 node / 1 material) ────────────────────────
    // Real data is written by uploadSceneIfDirty() before the first dispatch.
    VkDeviceSize nodeInit = sizeof(gpu::GPUNode);
    VkDeviceSize matInit  = sizeof(gpu::GPUMaterial);
    nodeBuffer_ = createAndMap(nodeInit, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                nodeMemory_, nodeMapped_);
    nodeCapacity_ = nodeInit;
    matBuffer_  = createAndMap(matInit,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                matMemory_,  matMapped_);
    matCapacity_ = matInit;

    // ── ComputePass ───────────────────────────────────────────────────────────
    computePass_.init(ctx_.getDevice());
    computePass_.writeDescriptors(ctx_.getDevice(),
        nodeBuffer_, nodeCapacity_,
        matBuffer_,  matCapacity_,
        uboBuffers_,
        storageImage_.getView());

    sync_.init(ctx_.getDevice(), kMaxFramesInFlight);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene construction
// ─────────────────────────────────────────────────────────────────────────────

void Application::initScene() {
    // ── Materials ─────────────────────────────────────────────────────────────
    int redId    = materials_.add("red",    scene::Material::diffuse({0.9f, 0.15f, 0.1f},  0.6f));
    int goldId   = materials_.add("gold",   scene::Material::metal  ({1.0f, 0.76f, 0.33f}, 0.2f));
    int whiteId  = materials_.add("white",  scene::Material::diffuse({0.9f, 0.9f,  0.9f},  0.9f));
    int glowId   = materials_.add("glow",   scene::Material::emitter({0.4f, 0.7f,  1.0f},  3.0f));

    auto redMat   = materials_.get(redId);
    auto goldMat  = materials_.get(goldId);
    auto whiteMat = materials_.get(whiteId);
    auto glowMat  = materials_.get(glowId);

    // ── Central smooth-union blob ─────────────────────────────────────────────
    auto sphere0 = scene::CSGBuilder::sphere("s0",
        scene::Transform::translate({0.0f, 0.3f, 0.0f}), 0.7f, redMat);

    auto sphere1 = scene::CSGBuilder::sphere("s1",
        scene::Transform::translate({0.9f, 0.0f, 0.3f}), 0.45f, goldMat);

    auto sphere2 = scene::CSGBuilder::sphere("s2",
        scene::Transform::translate({-0.7f, 0.1f, 0.5f}), 0.4f, goldMat);

    auto blob = scene::CSGBuilder::smoothU(
                    scene::CSGBuilder::smoothU(sphere0, sphere1, 0.35f),
                    sphere2, 0.25f);

    // ── Torus ring around it ──────────────────────────────────────────────────
    auto torus = scene::CSGBuilder::torus("ring",
        scene::Transform::fromAxisAngle({1,0,0}, 80.0f), 1.1f, 0.08f, goldMat);

    // ── Subtract a hole through the blob ─────────────────────────────────────
    auto holeCyl = scene::CSGBuilder::cappedCylinder("hole",
        scene::Transform::translate({0.0f, 0.0f, 0.0f}), 0.18f, 1.2f, whiteMat);

    auto blobWithHole = scene::CSGBuilder::smoothSub(blob, holeCyl, 0.05f);

    // ── Ground plane ──────────────────────────────────────────────────────────
    auto ground = scene::CSGBuilder::plane("ground",
        scene::Transform::identity(),
        {0.0f, 1.0f, 0.0f}, 1.2f, whiteMat);

    // ── Glowing sphere light stand-in ─────────────────────────────────────────
    auto lightOrb = scene::CSGBuilder::sphere("lightOrb",
        scene::Transform::translate({2.5f, 2.0f, -1.5f}), 0.2f, glowMat);

    // ── Assemble scene graph ──────────────────────────────────────────────────
    auto blobNode   = scene::SceneNode::make("blob");
    blobNode->csgRoot = scene::CSGBuilder::unite(blobWithHole, torus);
    blobNode->csgRoot->updateAABB();

    auto groundNode = scene::SceneNode::make("ground");
    groundNode->csgRoot = ground;

    auto lightNode  = scene::SceneNode::make("light");
    lightNode->csgRoot = lightOrb;

    sceneGraph_.addNode(blobNode);
    sceneGraph_.addNode(groundNode);
    sceneGraph_.addNode(lightNode);

    // Validate before flattening.
    scene::SceneValidator validator;
    if (!validator.validate(sceneGraph_)) {
        validator.report();
        throw std::runtime_error("Scene validation failed — see stderr.");
    }

    sceneDirty_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main loop
// ─────────────────────────────────────────────────────────────────────────────

void Application::mainLoop() {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        // Simple orbit controls: left/right arrows rotate the camera.
        if (glfwGetKey(window_, GLFW_KEY_LEFT)  == GLFW_PRESS) camTheta_ -= 0.02f;
        if (glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS) camTheta_ += 0.02f;
        if (glfwGetKey(window_, GLFW_KEY_UP)    == GLFW_PRESS) camPhi_    = std::max(0.05f, camPhi_ - 0.02f);
        if (glfwGetKey(window_, GLFW_KEY_DOWN)  == GLFW_PRESS) camPhi_    = std::min(1.55f, camPhi_ + 0.02f);

        drawFrame();
    }
    vkDeviceWaitIdle(ctx_.getDevice());
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene upload
// ─────────────────────────────────────────────────────────────────────────────

void Application::uploadSceneIfDirty() {
    if (!sceneDirty_) return;
    sceneDirty_ = false;

    flattener_.flatten(sceneGraph_, materials_);
    bvhRoot_ = flattener_.bvhRoot();

    const auto& nodes = flattener_.nodes();
    const auto& mats  = flattener_.gpuMaterials();

    VkDeviceSize nodeBytes = nodes.size() * sizeof(gpu::GPUNode);
    VkDeviceSize matBytes  = mats.size()  * sizeof(gpu::GPUMaterial);

    bool nodeRebuild = false;
    bool matRebuild  = false;

    if (nodeBytes > nodeCapacity_) {
        destroyBuffer(nodeBuffer_, nodeMemory_, nodeMapped_);
        VkDeviceSize newCap = std::max(nodeBytes, nodeCapacity_ + nodeCapacity_ / 2);
        nodeBuffer_  = createAndMap(newCap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     nodeMemory_, nodeMapped_);
        nodeCapacity_ = newCap;
        nodeRebuild  = true;
    }

    if (matBytes > matCapacity_) {
        destroyBuffer(matBuffer_, matMemory_, matMapped_);
        VkDeviceSize newCap = std::max(matBytes, matCapacity_ + matCapacity_ / 2);
        matBuffer_   = createAndMap(newCap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     matMemory_,  matMapped_);
        matCapacity_ = newCap;
        matRebuild   = true;
    }

    std::memcpy(nodeMapped_, nodes.data(), nodeBytes);
    std::memcpy(matMapped_,  mats.data(),  matBytes);

    if (nodeRebuild || matRebuild)
        computePass_.rebindBuffers(ctx_.getDevice(),
                                    nodeBuffer_, nodeCapacity_,
                                    matBuffer_,  matCapacity_);
}

// ─────────────────────────────────────────────────────────────────────────────
// UBO upload
// ─────────────────────────────────────────────────────────────────────────────

void Application::uploadUBO(int frameIndex, float time) {
    VkExtent2D ext = swapchain_.getExtent();

    // Orbit camera position.
    float x = camRadius_ * std::sin(camPhi_) * std::cos(camTheta_);
    float y = camRadius_ * std::cos(camPhi_);
    float z = camRadius_ * std::sin(camPhi_) * std::sin(camTheta_);
    glm::vec3 eye    = { x, y, z };
    glm::vec3 target = { 0.0f, 0.0f, 0.0f };
    glm::vec3 up     = { 0.0f, 1.0f, 0.0f };

    float aspect = static_cast<float>(ext.width) / static_cast<float>(ext.height);

    gpu::SceneUBO ubo{};
    ubo.view          = glm::lookAt(eye, target, up);
    ubo.projection    = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
    ubo.projection[1][1] *= -1.0f; // Vulkan Y flip
    ubo.invView       = glm::inverse(ubo.view);
    ubo.invProjection = glm::inverse(ubo.projection);
    ubo.cameraPos     = glm::vec4(eye, 1.0f);
    ubo.resolution    = { static_cast<float>(ext.width), static_cast<float>(ext.height), 0, 0 };
    ubo.time          = time;

    std::memcpy(uboMapped_[frameIndex], &ubo, sizeof(ubo));
}

// ─────────────────────────────────────────────────────────────────────────────
// Draw frame
// ─────────────────────────────────────────────────────────────────────────────

void Application::drawFrame() {
    VkDevice device = ctx_.getDevice();

    VkFence     fence         = sync_.inFlightFence();
    VkSemaphore imageAvailSem = sync_.imageAvailableSemaphore();

    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapchain_.getSwapchain(),
                                             UINT64_MAX, imageAvailSem,
                                             VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { handleResize(); return; }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swapchain image.");

    vkResetFences(device, 1, &fence);

    int frameIndex = sync_.currentFrame();
    float time     = static_cast<float>(glfwGetTime());

    uploadSceneIfDirty();
    uploadUBO(frameIndex, time);

    VkCommandBuffer cmd = commandPool_.getBuffer(frameIndex);
    vkResetCommandBuffer(cmd, 0);
    recordCommandBuffer(cmd, imageIndex, frameIndex);

    VkSemaphore          renderDoneSem = sync_.renderFinishedSemaphore();
    VkPipelineStageFlags waitStages[]  = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };

    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &imageAvailSem;
    si.pWaitDstStageMask    = waitStages;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &renderDoneSem;

    if (vkQueueSubmit(ctx_.getGraphicsQueue(), 1, &si, fence) != VK_SUCCESS)
        throw std::runtime_error("Failed to submit command buffer.");

    VkSwapchainKHR swapChains[] = { swapchain_.getSwapchain() };
    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderDoneSem;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = swapChains;
    pi.pImageIndices      = &imageIndex;

    result = vkQueuePresentKHR(ctx_.getPresentQueue(), &pi);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized_) {
        framebufferResized_ = false;
        handleResize();
    } else if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to present.");

    sync_.nextFrame();
}

// ─────────────────────────────────────────────────────────────────────────────
// Command buffer
// ─────────────────────────────────────────────────────────────────────────────

void Application::recordCommandBuffer(VkCommandBuffer cmd,
                                       uint32_t imageIndex,
                                       int frameIndex)
{
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin command buffer.");

    VkExtent2D extent    = swapchain_.getExtent();
    VkImage    swapImage = swapchain_.getImages()[imageIndex];
    VkImage    storeImg  = storageImage_.getImage();

    auto imgBarrier = [&](VkImage image,
                           VkImageLayout oldLayout,   VkImageLayout newLayout,
                           VkAccessFlags srcAccess,   VkAccessFlags dstAccess,
                           VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = oldLayout;
        b.newLayout           = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = image;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = srcAccess;
        b.dstAccessMask       = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    imgBarrier(storeImg,
        VK_IMAGE_LAYOUT_UNDEFINED,          VK_IMAGE_LAYOUT_GENERAL,
        0,                                   VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    computePass_.dispatch(cmd, frameIndex, extent, bvhRoot_);

    imgBarrier(storeImg,
        VK_IMAGE_LAYOUT_GENERAL,             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT,          VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    imgBarrier(swapImage,
        VK_IMAGE_LAYOUT_UNDEFINED,           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,                                    VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,   VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit blitRegion{};
    blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blitRegion.srcOffsets[0]  = { 0, 0, 0 };
    blitRegion.srcOffsets[1]  = { (int32_t)extent.width, (int32_t)extent.height, 1 };
    blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blitRegion.dstOffsets[0]  = { 0, 0, 0 };
    blitRegion.dstOffsets[1]  = { (int32_t)extent.width, (int32_t)extent.height, 1 };

    vkCmdBlitImage(cmd,
        storeImg,  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blitRegion, VK_FILTER_NEAREST);

    imgBarrier(swapImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT,          0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("Failed to end command buffer.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize
// ─────────────────────────────────────────────────────────────────────────────

void Application::handleResize() {
    swapchain_.recreate(ctx_, window_);
    storageImage_.recreate(ctx_, commandPool_.getPool(), swapchain_.getExtent());
    computePass_.updateStorageImage(ctx_.getDevice(), storageImage_.getView());
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer helpers
// ─────────────────────────────────────────────────────────────────────────────

VkBuffer Application::createAndMap(VkDeviceSize size, VkBufferUsageFlags usage,
                                     VkDeviceMemory& outMem, void*& outMapped)
{
    VkDevice device = ctx_.getDevice();
    VkBuffer buf;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("Application: failed to create buffer.");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buf, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = ctx_.findMemoryType(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &ai, nullptr, &outMem) != VK_SUCCESS)
        throw std::runtime_error("Application: failed to allocate buffer memory.");

    vkBindBufferMemory(device, buf, outMem, 0);
    vkMapMemory(device, outMem, 0, size, 0, &outMapped);
    return buf;
}

void Application::destroyBuffer(VkBuffer& buf, VkDeviceMemory& mem, void*& mapped) {
    VkDevice device = ctx_.getDevice();
    if (mapped) { vkUnmapMemory(device, mem); mapped = nullptr; }
    if (buf != VK_NULL_HANDLE) { vkDestroyBuffer(device, buf, nullptr); buf = VK_NULL_HANDLE; }
    if (mem != VK_NULL_HANDLE) { vkFreeMemory   (device, mem, nullptr); mem = VK_NULL_HANDLE; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cleanup
// ─────────────────────────────────────────────────────────────────────────────

void Application::cleanup() {
    VkDevice device = ctx_.getDevice();

    sync_.cleanup(device);
    computePass_.cleanup(device);
    storageImage_.destroy(device);

    destroyBuffer(nodeBuffer_, nodeMemory_, nodeMapped_);
    destroyBuffer(matBuffer_,  matMemory_,  matMapped_);
    for (int i = 0; i < kMaxFramesInFlight; ++i)
        destroyBuffer(uboBuffers_[i], uboMemories_[i], uboMapped_[i]);

    commandPool_.cleanup(device);
    swapchain_.cleanup(device);
    ctx_.cleanup();

    glfwDestroyWindow(window_);
    glfwTerminate();
}
