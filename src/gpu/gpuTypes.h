#pragma once

#include <cstdint>
#include <glm/glm.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// GPUTypes.h  —  Layer 3 (CPU↔GPU bridge)
//
// Every struct in this file has a matching GLSL declaration in gpu_types.glsl.
// The std430 layout rules that govern both sides:
//
//   float  → 4 bytes, aligned to 4
//   int    → 4 bytes, aligned to 4
//   vec2   → 8 bytes, aligned to 8
//   vec3   → 12 bytes, aligned to 16  (!)  padded to vec4 in practice
//   vec4   → 16 bytes, aligned to 16
//   mat4   → 64 bytes, aligned to 16
//
// We always use vec4 / alignas(16) rather than vec3 to avoid surprises.
// All sizes must be multiples of 16.
//
// NO scene-logic here.  NO Vulkan here.  Only plain data.
// ─────────────────────────────────────────────────────────────────────────────

namespace gpu {

// ─────────────────────────────────────────────────────────────────────────────
// GPUNode  —  one entry in the scene SSBO
//
// Represents EITHER a CSG tree node OR a BVH acceleration node — both live in
// the same flat array.  The op field distinguishes them.
//
// Layout (offset, size in bytes):
//   0   bboxMin      vec4  (xyz = AABB min, w = primType as float bits)
//   16  bboxMax      vec4  (xyz = AABB max, w = op      as float bits)
//   32  params0      vec4  primitive-specific data  (8 floats total across
//   p0+p1) 48  params1      vec4  primitive-specific data 64  leftChild    int
//   index into GPUNode[]; -1 = no left child / leaf 68  rightChild   int index
//   into GPUNode[]; -1 = no right child 72  materialId   int   index into
//   GPUMaterial[] 76  smoothK      float blending radius for smooth ops; 0 for
//   hard ops 80  invTransform mat4  pre-inverted TRS — brings world-space p
//   into local space 144 (total)
//
// primType and op are packed into the w components of bboxMin/bboxMax as
// bit-cast floats (intBitsToFloat / floatBitsToInt in GLSL) to avoid
// adding padding members that would break the stride.
//
// Params packing by primitive type (see SceneFlattener.cpp for CPU side,
// sdf_primitives.glsl for GPU side):
//
//   Sphere          p0.x           = radius
//   Box             p0.xyz         = halfExtents
//   RoundBox        p0.xyz         = halfExtents,  p0.w = radius
//   BoxFrame        p0.xyz         = halfExtents,  p0.w = edgeWidth
//   RoundedBoxFrame p0.xyz         = halfExtents,  p0.w = edgeWidth,  p1.x =
//   radius Torus           p0.x           = majorR,       p0.y = minorR
//   CappedTorus     p0.xy          = arc(sin,cos), p0.z = outerR,    p0.w =
//   innerR Link            p0.x           = halfLen,      p0.y = outerR, p0.z =
//   innerR CylinderInf     p0.xy          = centre,       p0.z = radius Cone
//   p0.xy          = sinCos,       p0.z = height Plane           p0.xyz =
//   normal,       p0.w = offset HexPrism        p0.x           = radius, p0.y =
//   height TriPrism        p0.x           = radius,       p0.y = height Capsule
//   p0.xyz         = pointA,       p0.w = radius,    p1.xyz = pointB
//   VerticalCapsule p0.x           = height,       p0.y = radius
//   CappedCylinder  p0.x           = radius,       p0.y = height
//   RoundedCylinder p0.x           = bodyR,        p0.y = roundR,    p0.z =
//   height CappedCone      p0.x           = height,       p0.y = r1, p0.z = r2
//   RoundCone       p0.x           = r1,           p0.y = r2,        p0.z =
//   height Ellipsoid       p0.xyz         = radii BvhBranch       (no params —
//   bbox + children only)
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) GPUNode {
  glm::vec4 bboxMin;      // xyz = AABB min,  w = primType (bit-cast int)
  glm::vec4 bboxMax;      // xyz = AABB max,  w = op       (bit-cast int)
  glm::vec4 params0;      // primitive-specific — see table above
  glm::vec4 params1;      // primitive-specific — see table above
  int32_t leftChild;      // array index; -1 = no child (leaf on this side)
  int32_t rightChild;     // array index; -1 = no child
  int32_t materialId;     // index into GPUMaterial SSBO; -1 = no material
  float smoothK;          // blending radius; 0 for non-smooth ops
  glm::mat4 invTransform; // pre-inverted TRS matrix (offset 80)
                          // — total struct size: 144 bytes
};

static_assert(sizeof(GPUNode) == 144,
              "GPUNode size mismatch — check std430 alignment rules.");
static_assert(alignof(GPUNode) == 16, "GPUNode alignment mismatch.");

// ─────────────────────────────────────────────────────────────────────────────
// GPUMaterial  —  one entry in the material SSBO
//
// Layout (32 bytes, stride 32):
//   0   albedo_roughness     vec4   xyz = albedo (linear),  w = roughness
//   16  metallic_emissive    vec4   x = metallic, y = emissive,
//                                   z = transparency, w = ior
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) GPUMaterial {
  glm::vec4 albedoRoughness;  // rgb + roughness
  glm::vec4 metallicEmissive; // metallic, emissive, transparency, ior
};

static_assert(sizeof(GPUMaterial) == 32, "GPUMaterial size mismatch.");

// ─────────────────────────────────────────────────────────────────────────────
// SceneUBO  —  per-frame uniform buffer
//
// Uploaded once per frame by UBOManager.
// invView and invProjection are pre-computed on the CPU so the shader never
// inverts a matrix — inversion inside a compute shader is executed by every
// thread in every workgroup, every frame.
//
// Layout (304 bytes):
//   0    view           mat4    64 bytes
//   64   projection     mat4    64 bytes
//   128  invView        mat4    64 bytes
//   192  invProjection  mat4    64 bytes
//   256  cameraPos      vec4    16 bytes  (w unused)
//   272  resolution     vec4    16 bytes  (xy = width/height, zw unused)
//   288  time           float    4 bytes
//   292  _pad[3]        float   12 bytes  (round to 16-byte alignment)
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) SceneUBO {
  glm::mat4 view;
  glm::mat4 projection;
  glm::mat4 invView;
  glm::mat4 invProjection;
  glm::vec4 cameraPos;
  glm::vec4 resolution;
  float time;
  float _pad[3];
};

static_assert(sizeof(SceneUBO) == 304, "SceneUBO size mismatch.");

// ─────────────────────────────────────────────────────────────────────────────
// GPUSceneBuffer  —  the complete upload payload
//
// Handed to SSBOManager for upload.  This is a view into CPU-side vectors,
// not an owning type.
// ─────────────────────────────────────────────────────────────────────────────

struct GPUSceneBuffer {
  const GPUNode *nodes;
  uint32_t nodeCount;
  const GPUMaterial *materials;
  uint32_t materialCount;
  int32_t bvhRootIndex; // index of the BVH root node in `nodes`
};

// ─────────────────────────────────────────────────────────────────────────────
// Packing helpers
//
// Pack a uint32_t into a float's bit pattern without violating strict aliasing.
// These mirror GLSL's floatBitsToInt / intBitsToFloat.
// ─────────────────────────────────────────────────────────────────────────────

inline float packUint(uint32_t v) {
  float f;
  static_assert(sizeof(float) == sizeof(uint32_t));
  __builtin_memcpy(&f, &v, sizeof(float));
  return f;
}

inline uint32_t unpackUint(float f) {
  uint32_t v;
  __builtin_memcpy(&v, &f, sizeof(uint32_t));
  return v;
}

} // namespace gpu
