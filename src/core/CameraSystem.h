#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// CameraSystem
//
// Holds a list of named static cameras.  No orbit, no movement.
// Switch with next() / prev() or setIndex().
//
// The active camera produces a view matrix and projection parameters
// that Application feeds into the SceneUBO each frame.
// ─────────────────────────────────────────────────────────────────────────────

struct Camera {
  std::string name;
  glm::vec3 position;
  glm::vec3 target;
  glm::vec3 up = {0.0f, 1.0f, 0.0f};
  float fovDeg = 60.0f;
};

class CameraSystem {
public:
  void add(Camera cam) { cameras_.push_back(std::move(cam)); }

  void add(const std::string &name, glm::vec3 position, glm::vec3 target,
           float fovDeg = 60.0f) {
    cameras_.push_back({name, position, target, {0, 1, 0}, fovDeg});
  }

  // ── Navigation ────────────────────────────────────────────────────────────
  void next() {
    if (!cameras_.empty())
      index_ = (index_ + 1) % cameras_.size();
  }
  void prev() {
    if (!cameras_.empty())
      index_ = (index_ + cameras_.size() - 1) % cameras_.size();
  }
  void setIndex(int i) { index_ = static_cast<size_t>(i) % cameras_.size(); }

  // ── Active camera ─────────────────────────────────────────────────────────
  const Camera &active() const {
    if (cameras_.empty())
      throw std::runtime_error("CameraSystem: no cameras registered.");
    return cameras_[index_];
  }

  int currentIndex() const { return static_cast<int>(index_); }
  int count() const { return static_cast<int>(cameras_.size()); }
  const std::string &currentName() const { return active().name; }

  // ── Matrix output ─────────────────────────────────────────────────────────
  glm::mat4 viewMatrix() const {
    const auto &c = active();
    return glm::lookAt(c.position, c.target, c.up);
  }

  glm::mat4 projectionMatrix(float aspect) const {
    glm::mat4 proj =
        glm::perspective(glm::radians(active().fovDeg), aspect, 0.1f, 200.0f);
    proj[1][1] *= -1.0f; // Vulkan Y flip
    return proj;
  }

  glm::vec3 position() const { return active().position; }

private:
  std::vector<Camera> cameras_;
  size_t index_ = 0;
};
