#pragma once

#include <memory>
#include <string>

#include "./node.h"

// ─────────────────────────────────────────────────────────────────────────────
// CSGBuilder.h  —  Layer 1
//
// Static factory API.  All methods return a CSGNode::Ptr so the caller can
// immediately compose with the boolean operators:
//
//   auto leg = CSGBuilder::cylinder("leg", t, 0.05f, 0.4f, mat);
//   auto knob = CSGBuilder::sphere("knob", tipTransform, 0.07f, mat);
//   auto legWithKnob = CSGBuilder::smoothU(leg, knob, 0.04f);
//
// The string `name` parameter is a debug label stored on the node.
// Pass "" for anonymous nodes.
//
// No Vulkan.  No scene graph.  No GPU types.
// ─────────────────────────────────────────────────────────────────────────────

namespace scene {

struct CSGBuilder {

  // ── Primitives ────────────────────────────────────────────────────────────

  static CSGNode::Ptr sphere(const std::string &name, const Transform &t,
                             float radius, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = SphereParams{radius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr box(const std::string &name, const Transform &t,
                          glm::vec3 halfExtents, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = BoxParams{halfExtents};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr roundBox(const std::string &name, const Transform &t,
                               glm::vec3 halfExtents, float radius,
                               const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = RoundBoxParams{halfExtents, radius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr boxFrame(const std::string &name, const Transform &t,
                               glm::vec3 halfExtents, float edgeWidth,
                               const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = BoxFrameParams{halfExtents, edgeWidth};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr roundedBoxFrame(const std::string &name,
                                      const Transform &t, glm::vec3 halfExtents,
                                      float edgeWidth, float radius,
                                      const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = RoundedBoxFrameParams{halfExtents, edgeWidth, radius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr torus(const std::string &name, const Transform &t,
                            float majorRadius, float minorRadius,
                            const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = TorusParams{majorRadius, minorRadius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr cappedTorus(const std::string &name, const Transform &t,
                                  float halfAngleDeg, float outerR,
                                  float innerR, const Material &mat) {
    auto n = leaf(name, t, mat);
    float a = glm::radians(halfAngleDeg);
    n->prim = CappedTorusParams{{std::sin(a), std::cos(a)}, outerR, innerR};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr link(const std::string &name, const Transform &t,
                           float halfLength, float outerR, float innerR,
                           const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = LinkParams{halfLength, outerR, innerR};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr cylinderInfinite(const std::string &name,
                                       const Transform &t, glm::vec2 centre,
                                       float radius, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = CylinderInfiniteParams{centre, radius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr cone(const std::string &name, const Transform &t,
                           float halfAngleDeg, float height,
                           const Material &mat) {
    auto n = leaf(name, t, mat);
    float a = glm::radians(halfAngleDeg);
    n->prim = ConeParams{{std::sin(a), std::cos(a)}, height};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr plane(const std::string &name, const Transform &t,
                            glm::vec3 normal, float offset,
                            const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = PlaneParams{glm::normalize(normal), offset};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr hexPrism(const std::string &name, const Transform &t,
                               float radius, float height,
                               const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = HexPrismParams{radius, height};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr triPrism(const std::string &name, const Transform &t,
                               float radius, float height,
                               const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = TriPrismParams{radius, height};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr capsule(const std::string &name, const Transform &t,
                              glm::vec3 pointA, glm::vec3 pointB, float radius,
                              const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = CapsuleParams{pointA, pointB, radius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr verticalCapsule(const std::string &name,
                                      const Transform &t, float height,
                                      float radius, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = VerticalCapsuleParams{height, radius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr cappedCylinder(const std::string &name,
                                     const Transform &t, float radius,
                                     float height, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = CappedCylinderParams{radius, height};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr roundedCylinder(const std::string &name,
                                      const Transform &t, float bodyRadius,
                                      float roundRadius, float height,
                                      const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = RoundedCylinderParams{bodyRadius, roundRadius, height};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr cappedCone(const std::string &name, const Transform &t,
                                 float height, float bottomRadius,
                                 float topRadius, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = CappedConeParams{height, bottomRadius, topRadius};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr roundCone(const std::string &name, const Transform &t,
                                float bottomRadius, float topRadius,
                                float height, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = RoundConeParams{bottomRadius, topRadius, height};
    n->updateAABB();
    return n;
  }

  static CSGNode::Ptr ellipsoid(const std::string &name, const Transform &t,
                                glm::vec3 radii, const Material &mat) {
    auto n = leaf(name, t, mat);
    n->prim = EllipsoidParams{radii};
    n->updateAABB();
    return n;
  }

  // ── Boolean operators ─────────────────────────────────────────────────────

  static CSGNode::Ptr unite(CSGNode::Ptr a, CSGNode::Ptr b) {
    return branch("", CSGOp::Union, 0.0f, std::move(a), std::move(b));
  }

  static CSGNode::Ptr subtract(CSGNode::Ptr a, CSGNode::Ptr b) {
    return branch("", CSGOp::Subtraction, 0.0f, std::move(a), std::move(b));
  }

  static CSGNode::Ptr intersect(CSGNode::Ptr a, CSGNode::Ptr b) {
    return branch("", CSGOp::Intersection, 0.0f, std::move(a), std::move(b));
  }

  static CSGNode::Ptr smoothU(CSGNode::Ptr a, CSGNode::Ptr b, float k) {
    return branch("", CSGOp::SmoothUnion, k, std::move(a), std::move(b));
  }

  static CSGNode::Ptr smoothSub(CSGNode::Ptr a, CSGNode::Ptr b, float k) {
    return branch("", CSGOp::SmoothSubtraction, k, std::move(a), std::move(b));
  }

  static CSGNode::Ptr smoothInt(CSGNode::Ptr a, CSGNode::Ptr b, float k) {
    return branch("", CSGOp::SmoothIntersection, k, std::move(a), std::move(b));
  }

  // Generic combine — use when the op is determined at runtime.
  static CSGNode::Ptr combine(CSGOp op, CSGNode::Ptr a, CSGNode::Ptr b,
                              float k = 0.0f) {
    return branch("", op, k, std::move(a), std::move(b));
  }

private:
  // ── Internal helpers ──────────────────────────────────────────────────────

  static CSGNode::Ptr leaf(const std::string &name, const Transform &t,
                           const Material &mat) {
    auto n = std::make_shared<CSGNode>();
    n->name = name;
    n->op = CSGOp::Leaf;
    n->transform = t;
    n->material = mat;
    return n;
  }

  static CSGNode::Ptr branch(const std::string &name, CSGOp op, float k,
                             CSGNode::Ptr left, CSGNode::Ptr right) {
    auto n = std::make_shared<CSGNode>();
    n->name = name;
    n->op = op;
    n->smoothK = k;
    n->left = std::move(left);
    n->right = std::move(right);
    n->updateAABB();
    return n;
  }
};

} // namespace scene
