#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Transform.h  —  Layer 0
//
// Wraps position + quaternion rotation + scale into one struct.
// toMatrix()    — compose a column-major mat4 (TRS order: scale → rotate →
// translate) invMatrix()   — pre-invert for GPU upload; avoids doing this in
// the shader lerp()        — component-wise linear interpolation (position +
// scale) slerp()       — correct quaternion slerp for rotation
//
// No Vulkan.  No CSG.  No scene graph.
// ─────────────────────────────────────────────────────────────────────────────

namespace scene {

struct Transform {
  glm::vec3 position = {0.0f, 0.0f, 0.0f};
  glm::quat rotation = glm::identity<glm::quat>();
  glm::vec3 scale = {1.0f, 1.0f, 1.0f};

  // ── Construction helpers ──────────────────────────────────────────────────

  static Transform identity() { return {}; }

  static Transform translate(glm::vec3 pos) {
    Transform t;
    t.position = pos;
    return t;
  }

  static Transform rotate(glm::vec3 eulerDegrees) {
    Transform t;
    t.rotation = glm::quat(glm::radians(eulerDegrees));
    return t;
  }

  static Transform fromAxisAngle(glm::vec3 axis, float angleDeg) {
    Transform t;
    t.rotation = glm::angleAxis(glm::radians(angleDeg), glm::normalize(axis));
    return t;
  }

  static Transform uniform(float s) {
    Transform t;
    t.scale = {s, s, s};
    return t;
  }

  // ── Matrix output ─────────────────────────────────────────────────────────

  // TRS: Scale first, then rotate, then translate.
  // This matches the GPU convention where local-space points are transformed
  // as:  worldPos = T * R * S * localPos
  glm::mat4 toMatrix() const {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 R = glm::toMat4(rotation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    return T * R * S;
  }

  // Pre-inverted for the GPU.  The SDF evaluator applies invMatrix to the
  // ray-march point, bringing it into local primitive space without needing
  // to invert per pixel.
  glm::mat4 invMatrix() const { return glm::inverse(toMatrix()); }

  // 3×3 rotation-only matrix, used for normal transformation.
  // If scale is non-uniform, normals must use the inverse-transpose instead —
  // see normalMatrix().
  glm::mat3 rotationMatrix() const { return glm::mat3(glm::toMat4(rotation)); }

  glm::mat3 normalMatrix() const {
    return glm::transpose(glm::inverse(glm::mat3(toMatrix())));
  }

  // ── Interpolation ─────────────────────────────────────────────────────────

  // t = 0 → this,  t = 1 → other.
  // Position and scale use glm::mix (linear); rotation uses glm::slerp.
  Transform lerp(const Transform &other, float t) const {
    Transform result;
    result.position = glm::mix(position, other.position, t);
    result.rotation = glm::slerp(rotation, other.rotation, t);
    result.scale = glm::mix(scale, other.scale, t);
    return result;
  }

  // Compose: apply `child` in the local space of `this`.
  // Equivalent to parent.toMatrix() * child.toMatrix(), but stays in
  // Transform form so callers don't have to decompose a mat4.
  Transform compose(const Transform &child) const {
    Transform result;
    result.position =
        position + glm::mat3(glm::toMat4(rotation)) * (scale * child.position);
    result.rotation = rotation * child.rotation;
    result.scale = scale * child.scale;
    return result;
  }

  bool operator==(const Transform &o) const {
    return position == o.position && rotation == o.rotation && scale == o.scale;
  }
  bool operator!=(const Transform &o) const { return !(*this == o); }
};

} // namespace scene
