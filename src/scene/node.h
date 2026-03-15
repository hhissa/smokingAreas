#pragma once

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "./material.h"
#include "./primitives.h"
#include "./transform.h"

// ─────────────────────────────────────────────────────────────────────────────
// CSGNode.h  —  Layer 1
//
// Binary tree node.  A node is either:
//   Leaf   — holds a primitive (PrimParams) + transform + material
//   Branch — holds a CSGOp combining left and right children
//
// AABB is cached here and recomputed when the tree is modified.  The flattener
// reads it for BVH construction without traversing the tree again.
//
// Ownership model: shared_ptr so that instancing (SceneGraph) can share
// subtrees between SceneNodes without deep-copying.
//
// No Vulkan.  No scene graph.  No GPU types.
// ─────────────────────────────────────────────────────────────────────────────

namespace scene {

// ── CSG operation enum
// ──────────────────────────────────────────────────────── Integer values are
// written into GPUNode.op — keep them stable.
enum class CSGOp : uint32_t {
  Leaf = 0, // terminal — evaluates prim
  Union = 1,
  Subtraction = 2,
  Intersection = 3,
  SmoothUnion = 4,
  SmoothSubtraction = 5,
  SmoothIntersection = 6,
  BvhBranch = 7, // spatial acceleration node — not a CSG op
                 // leftChild/rightChild point to GPUNode array indices
                 // GPU traverses bbox first, only descends on hit
};

// ── Axis-aligned bounding box
// ─────────────────────────────────────────────────
struct AABB {
  glm::vec3 min = {+1e30f, +1e30f, +1e30f};
  glm::vec3 max = {-1e30f, -1e30f, -1e30f};

  bool isValid() const { return min.x <= max.x; }

  glm::vec3 centre() const { return (min + max) * 0.5f; }
  glm::vec3 extent() const { return max - min; }
  float surfaceArea() const {
    glm::vec3 e = extent();
    return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
  }

  // Expand to enclose another AABB.
  AABB merge(const AABB &o) const {
    return {glm::min(min, o.min), glm::max(max, o.max)};
  }

  // Expand by a scalar margin (used for smooth ops where the field bleeds
  // outside the geometric boundary by smoothK).
  AABB expand(float margin) const {
    return {min - glm::vec3(margin), max + glm::vec3(margin)};
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// CSGNode
// ─────────────────────────────────────────────────────────────────────────────

struct CSGNode {
  using Ptr = std::shared_ptr<CSGNode>;

  // ── Identity ──────────────────────────────────────────────────────────────
  std::string name; // optional debug label

  // ── Topology ─────────────────────────────────────────────────────────────
  CSGOp op = CSGOp::Leaf;
  float smoothK =
      0.0f; // only used by SmoothUnion / SmoothSubtraction / SmoothIntersection

  // ── Leaf data (only valid when op == Leaf) ────────────────────────────────
  PrimParams prim = SphereParams{};
  Transform transform = Transform::identity();
  Material material;

  // ── Branch data (only valid when op != Leaf) ──────────────────────────────
  Ptr left;
  Ptr right;

  // ── Cached AABB (updated by updateAABB()) ─────────────────────────────────
  AABB bbox;

  // ── Queries ───────────────────────────────────────────────────────────────
  bool isLeaf() const { return op == CSGOp::Leaf; }

  // Recursively recompute the AABB bottom-up.
  // Call after any structural change (node add, transform edit, etc.).
  void updateAABB();

private:
  AABB computeLeafAABB() const;
};

// ─────────────────────────────────────────────────────────────────────────────
// AABB computation
// ─────────────────────────────────────────────────────────────────────────────

inline AABB CSGNode::computeLeafAABB() const {
  // Compute a conservative world-space AABB by transforming the 8 corners
  // of the local-space primitive AABB through the node's transform.
  auto localAABB = std::visit(
      [](const auto &p) -> AABB {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, SphereParams>) {
          return {glm::vec3(-p.radius), glm::vec3(p.radius)};
        } else if constexpr (std::is_same_v<T, BoxParams>) {
          return {-p.halfExtents, p.halfExtents};
        } else if constexpr (std::is_same_v<T, RoundBoxParams>) {
          glm::vec3 r(p.radius);
          return {-p.halfExtents - r, p.halfExtents + r};
        } else if constexpr (std::is_same_v<T, BoxFrameParams>) {
          return {-p.halfExtents, p.halfExtents};
        } else if constexpr (std::is_same_v<T, RoundedBoxFrameParams>) {
          glm::vec3 r(p.radius);
          return {-p.halfExtents - r, p.halfExtents + r};
        } else if constexpr (std::is_same_v<T, TorusParams>) {
          float r = p.majorRadius + p.minorRadius;
          return {glm::vec3(-r, -p.minorRadius, -r),
                  glm::vec3(r, p.minorRadius, r)};
        } else if constexpr (std::is_same_v<T, CappedTorusParams>) {
          float r = p.outerR + p.innerR;
          return {glm::vec3(-r, -p.innerR, -r), glm::vec3(r, p.innerR, r)};
        } else if constexpr (std::is_same_v<T, LinkParams>) {
          float r = p.outerR + p.innerR;
          return {glm::vec3(-r, -p.halfLength - p.innerR, -p.innerR),
                  glm::vec3(r, p.halfLength + p.innerR, p.innerR)};
        } else if constexpr (std::is_same_v<T, CylinderInfiniteParams>) {
          // Infinite along Y — use a large sentinel.
          return {glm::vec3(-p.radius, -1e10f, -p.radius),
                  glm::vec3(p.radius, 1e10f, p.radius)};
        } else if constexpr (std::is_same_v<T, ConeParams>) {
          float baseR = p.sinCos.x / p.sinCos.y * p.height;
          return {glm::vec3(-baseR, 0.0f, -baseR),
                  glm::vec3(baseR, p.height, baseR)};
        } else if constexpr (std::is_same_v<T, PlaneParams>) {
          // 1e4f is large enough to cover any reasonable scene without
          // overflowing to inf when multiplied together in surfaceArea().
          return {glm::vec3(-1e4f) - glm::vec3(p.normal) * p.offset,
                  glm::vec3(1e4f) - glm::vec3(p.normal) * p.offset};
        } else if constexpr (std::is_same_v<T, HexPrismParams>) {
          return {glm::vec3(-p.radius, -p.height, -p.radius),
                  glm::vec3(p.radius, p.height, p.radius)};
        } else if constexpr (std::is_same_v<T, TriPrismParams>) {
          return {glm::vec3(-p.radius, -p.height, -p.radius),
                  glm::vec3(p.radius, p.height, p.radius)};
        } else if constexpr (std::is_same_v<T, CapsuleParams>) {
          glm::vec3 lo = glm::min(p.pointA, p.pointB) - glm::vec3(p.radius);
          glm::vec3 hi = glm::max(p.pointA, p.pointB) + glm::vec3(p.radius);
          return {lo, hi};
        } else if constexpr (std::is_same_v<T, VerticalCapsuleParams>) {
          return {glm::vec3(-p.radius, -p.radius, -p.radius),
                  glm::vec3(p.radius, p.height + p.radius, p.radius)};
        } else if constexpr (std::is_same_v<T, CappedCylinderParams>) {
          return {glm::vec3(-p.radius, -p.height, -p.radius),
                  glm::vec3(p.radius, p.height, p.radius)};
        } else if constexpr (std::is_same_v<T, RoundedCylinderParams>) {
          float r = p.bodyRadius + p.roundRadius;
          float h = p.height + p.roundRadius;
          return {glm::vec3(-r, -h, -r), glm::vec3(r, h, r)};
        } else if constexpr (std::is_same_v<T, CappedConeParams>) {
          float maxR = std::max(p.bottomRadius, p.topRadius);
          return {glm::vec3(-maxR, 0.0f, -maxR),
                  glm::vec3(maxR, p.height, maxR)};
        } else if constexpr (std::is_same_v<T, RoundConeParams>) {
          float maxR = std::max(p.bottomRadius, p.topRadius);
          return {glm::vec3(-maxR, 0.0f, -maxR),
                  glm::vec3(maxR, p.height, maxR)};
        } else if constexpr (std::is_same_v<T, EllipsoidParams>) {
          return {-p.radii, p.radii};
        } else {
          return {};
        }
      },
      prim);

  // Transform the 8 local corners into world space.
  glm::mat4 m = transform.toMatrix();
  glm::vec3 corners[8];
  for (int i = 0; i < 8; ++i) {
    glm::vec3 c = {(i & 1) ? localAABB.max.x : localAABB.min.x,
                   (i & 2) ? localAABB.max.y : localAABB.min.y,
                   (i & 4) ? localAABB.max.z : localAABB.min.z};
    corners[i] = glm::vec3(m * glm::vec4(c, 1.0f));
  }

  AABB world;
  world.min = world.max = corners[0];
  for (int i = 1; i < 8; ++i) {
    world.min = glm::min(world.min, corners[i]);
    world.max = glm::max(world.max, corners[i]);
  }
  return world;
}

inline void CSGNode::updateAABB() {
  if (isLeaf()) {
    bbox = computeLeafAABB();
    return;
  }

  AABB childBBox;

  if (left) {
    left->updateAABB();
    childBBox = left->bbox;
  }
  if (right) {
    right->updateAABB();
    childBBox =
        childBBox.isValid() ? childBBox.merge(right->bbox) : right->bbox;
  }

  // Smooth ops bleed outside the geometric boundary by smoothK.
  if (op == CSGOp::SmoothUnion || op == CSGOp::SmoothSubtraction ||
      op == CSGOp::SmoothIntersection) {
    childBBox = childBBox.expand(smoothK);
  }

  bbox = childBBox;
}

} // namespace scene
  // namespace scene
