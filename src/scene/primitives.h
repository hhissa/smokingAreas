#pragma once

#include <glm/glm.hpp>
#include <variant>

namespace scene {

// ── Primitive parameter structs
// ───────────────────────────────────────────────

struct SphereParams {
  float radius = 0.5f;
};

struct BoxParams {
  glm::vec3 halfExtents = {0.5f, 0.5f, 0.5f};
};

struct RoundBoxParams {
  glm::vec3 halfExtents = {0.5f, 0.5f, 0.5f};
  float radius = 0.05f;
};

struct BoxFrameParams {
  glm::vec3 halfExtents = {0.5f, 0.5f, 0.5f};
  float edgeWidth = 0.05f;
};

struct RoundedBoxFrameParams {
  glm::vec3 halfExtents = {0.5f, 0.5f, 0.5f};
  float edgeWidth = 0.05f;
  float radius = 0.02f;
};

struct TorusParams {
  float majorRadius = 0.5f;  // ring centre to tube centre
  float minorRadius = 0.15f; // tube radius
};

struct CappedTorusParams {
  glm::vec2 arc = {0.866f, 0.5f}; // (sin θ, cos θ) — half-angle of cap
  float outerR = 0.5f;
  float innerR = 0.15f;
};

struct LinkParams {
  float halfLength = 0.3f; // half-length of the straight segment
  float outerR = 0.4f;
  float innerR = 0.1f;
};

struct CylinderInfiniteParams {
  glm::vec2 centre = {0.0f, 0.0f}; // XZ offset
  float radius = 0.5f;
};

struct ConeParams {
  glm::vec2 sinCos = {0.866f, 0.5f}; // (sin α, cos α) of half-angle
  float height = 1.0f;
};

struct PlaneParams {
  glm::vec3 normal = {0.0f, 1.0f, 0.0f};
  float offset = 0.0f;
};

struct HexPrismParams {
  float radius = 0.5f;
  float height = 0.3f;
};

struct TriPrismParams {
  float radius = 0.5f;
  float height = 0.3f;
};

struct CapsuleParams {
  glm::vec3 pointA = {0.0f, 0.5f, 0.0f};
  glm::vec3 pointB = {0.0f, -0.5f, 0.0f};
  float radius = 0.15f;
};

struct VerticalCapsuleParams {
  float height = 1.0f;
  float radius = 0.15f;
};

struct CappedCylinderParams {
  float radius = 0.5f;
  float height = 0.5f;
};

struct RoundedCylinderParams {
  float bodyRadius = 0.4f;
  float roundRadius = 0.05f;
  float height = 0.5f;
};

struct CappedConeParams {
  float height = 1.0f;
  float bottomRadius = 0.5f;
  float topRadius = 0.1f;
};

struct RoundConeParams {
  float bottomRadius = 0.5f;
  float topRadius = 0.1f;
  float height = 1.0f;
};

struct EllipsoidParams {
  glm::vec3 radii = {0.5f, 0.3f, 0.4f};
};

// ── Discriminator enum
// ──────────────────────────────────────────────────────── Matches the order of
// types in PrimParams below. The GPU flattener uses this to fill
// GPUNode.primType.

enum class PrimType : uint32_t {
  Sphere = 0,
  Box,
  RoundBox,
  BoxFrame,
  RoundedBoxFrame,
  Torus,
  CappedTorus,
  Link,
  CylinderInfinite,
  Cone,
  Plane,
  HexPrism,
  TriPrism,
  Capsule,
  VerticalCapsule,
  CappedCylinder,
  RoundedCylinder,
  CappedCone,
  RoundCone,
  Ellipsoid,
};

// ── PrimParams — the canonical primitive description
// ────────────────────────── std::visit with overloaded lambdas is the
// idiomatic access pattern:
//
//   std::visit(overloaded{
//       [](const SphereParams& s) { /* use s.radius */ },
//       [](const BoxParams& b)    { /* use b.halfExtents */ },
//       ...
//   }, node.prim);

using PrimParams = std::variant<
    SphereParams, BoxParams, RoundBoxParams, BoxFrameParams,
    RoundedBoxFrameParams, TorusParams, CappedTorusParams, LinkParams,
    CylinderInfiniteParams, ConeParams, PlaneParams, HexPrismParams,
    TriPrismParams, CapsuleParams, VerticalCapsuleParams, CappedCylinderParams,
    RoundedCylinderParams, CappedConeParams, RoundConeParams, EllipsoidParams>;

// Returns the PrimType discriminator for any PrimParams value.
// Used by the flattener to write GPUNode.primType.
inline PrimType primTypeOf(const PrimParams &p) {
  return static_cast<PrimType>(p.index());
}

} // namespace scene
