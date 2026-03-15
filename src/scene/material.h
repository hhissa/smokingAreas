#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Material.h  —  Layer 0
//
// Material holds every PBR parameter the shader needs.  The GPU only ever sees
// a materialId (int index into the material SSBO); the CPU side owns the full
// struct.
//
// MaterialRegistry maps string names → indices.  Use it to look up materials
// by name and to iterate the flat array for upload.
//
// No Vulkan.  No CSG.  No rendering calls.
// ─────────────────────────────────────────────────────────────────────────────

namespace scene {

struct Material {
  // ── PBR base ──────────────────────────────────────────────────────────────
  glm::vec3 albedo = {0.8f, 0.8f, 0.8f}; // linear-space base colour
  float roughness = 0.5f;                // 0 = mirror, 1 = fully diffuse
  float metallic = 0.0f;                 // 0 = dielectric, 1 = metal
  float emissive = 0.0f;                 // emissive scale (multiplies albedo)
  float transparency = 0.0f;             // 0 = opaque, 1 = fully transparent
  float ior = 1.45f;                     // index of refraction (glass ≈ 1.5)

  // ── Runtime identity ──────────────────────────────────────────────────────
  // Assigned by MaterialRegistry::add().  -1 = not yet registered.
  int materialId = -1;

  // ── Named constructors ────────────────────────────────────────────────────

  static Material diffuse(glm::vec3 albedo, float roughness = 0.8f) {
    Material m;
    m.albedo = albedo;
    m.roughness = roughness;
    return m;
  }

  static Material metal(glm::vec3 albedo, float roughness = 0.2f) {
    Material m;
    m.albedo = albedo;
    m.metallic = 1.0f;
    m.roughness = roughness;
    return m;
  }

  static Material glass(glm::vec3 tint = {1.0f, 1.0f, 1.0f}, float ior = 1.5f) {
    Material m;
    m.albedo = tint;
    m.roughness = 0.0f;
    m.transparency = 1.0f;
    m.ior = ior;
    return m;
  }

  static Material emitter(glm::vec3 colour, float intensity = 1.0f) {
    Material m;
    m.albedo = colour;
    m.roughness = 1.0f;
    m.emissive = intensity;
    return m;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// MaterialRegistry
//
// Single source of truth for all materials in a scene.
// add()       — register a named material; returns its index
// get()       — look up by name; throws if not found
// materials() — flat vector ordered by index, ready to memcpy into the GPU SSBO
// ─────────────────────────────────────────────────────────────────────────────

class MaterialRegistry {
public:
  // Register a material.  Returns the index that should be stored in
  // CSGNode::material.materialId.  Duplicate names overwrite the old entry.
  int add(const std::string &name, Material mat) {
    auto it = nameToIndex_.find(name);
    if (it != nameToIndex_.end()) {
      int idx = it->second;
      mat.materialId = idx;
      materials_[idx] = mat;
      return idx;
    }
    int idx = static_cast<int>(materials_.size());
    mat.materialId = idx;
    materials_.push_back(mat);
    nameToIndex_[name] = idx;
    return idx;
  }

  // Look up by name.  Throws std::out_of_range if not registered.
  const Material &get(const std::string &name) const {
    auto it = nameToIndex_.find(name);
    if (it == nameToIndex_.end())
      throw std::out_of_range("MaterialRegistry: unknown material '" + name +
                              "'");
    return materials_[it->second];
  }

  // Look up by index.
  const Material &get(int idx) const {
    return materials_.at(static_cast<size_t>(idx));
  }

  bool contains(const std::string &name) const {
    return nameToIndex_.count(name) > 0;
  }

  // Ordered flat array — hand this to the GPU upload layer.
  const std::vector<Material> &materials() const { return materials_; }

  int count() const { return static_cast<int>(materials_.size()); }

  void clear() {
    materials_.clear();
    nameToIndex_.clear();
  }

private:
  std::vector<Material> materials_;
  std::unordered_map<std::string, int> nameToIndex_;
};

} // namespace scene
