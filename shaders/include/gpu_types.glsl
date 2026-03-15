// gpu_types.glsl
//
// GLSL-side declarations of every struct in GPUTypes.h.
// Include this before sdf_primitives.glsl in the raymarch shader.
//
// std430 layout rules are identical to GPUTypes.h.
// Sizes must be verified against the C++ static_asserts.

// ── GPUNode ───────────────────────────────────────────────────────────────────
// One entry in the scene SSBO.  Represents either a CSG node or a BVH node.
// Total size: 144 bytes.  Stride in the SSBO array: 144 bytes.

struct GPUNode {
  vec4 bboxMin; // xyz = AABB min,  w = primType (floatBitsToInt)
  vec4 bboxMax; // xyz = AABB max,  w = op       (floatBitsToInt)
  vec4 params0; // primitive data — see GPUTypes.h packing table
  vec4 params1; // primitive data
  int leftChild; // array index; -1 = no child
  int rightChild; // array index; -1 = no child
  int materialId; // index into materials[]; -1 = unset
  float smoothK; // blending radius; 0 for hard ops
  mat4 invTransform; // pre-inverted TRS — transforms world p into local space
};

// ── GPUMaterial ───────────────────────────────────────────────────────────────
// Total size: 32 bytes.

struct GPUMaterial {
  vec4 albedoRoughness; // rgb = albedo (linear),  w = roughness
  vec4 metallicEmissive; // x = metallic, y = emissive, z = transparency, w = ior
};

// ── Op constants ─────────────────────────────────────────────────────────────
// Must match CSGOp enum in CSGNode.h exactly.

const int OP_LEAF = 0;
const int OP_UNION = 1;
const int OP_SUBTRACTION = 2;
const int OP_INTERSECTION = 3;
const int OP_SMOOTH_UNION = 4;
const int OP_SMOOTH_SUBTRACTION = 5;
const int OP_SMOOTH_INTERSECTION = 6;
const int OP_BVH_BRANCH = 7;

// ── PrimType constants ────────────────────────────────────────────────────────
// Must match PrimType enum in Primitives.h exactly.

const int PRIM_SPHERE = 0;
const int PRIM_BOX = 1;
const int PRIM_ROUND_BOX = 2;
const int PRIM_BOX_FRAME = 3;
const int PRIM_ROUNDED_BOX_FRAME = 4;
const int PRIM_TORUS = 5;
const int PRIM_CAPPED_TORUS = 6;
const int PRIM_LINK = 7;
const int PRIM_CYLINDER_INF = 8;
const int PRIM_CONE = 9;
const int PRIM_PLANE = 10;
const int PRIM_HEX_PRISM = 11;
const int PRIM_TRI_PRISM = 12;
const int PRIM_CAPSULE = 13;
const int PRIM_VERTICAL_CAPSULE = 14;
const int PRIM_CAPPED_CYLINDER = 15;
const int PRIM_ROUNDED_CYLINDER = 16;
const int PRIM_CAPPED_CONE = 17;
const int PRIM_ROUND_CONE = 18;
const int PRIM_ELLIPSOID = 19;

// ── SSBO bindings ─────────────────────────────────────────────────────────────
// Binding slots must match DescriptorManager's layout (Phase 5).

layout(std430, binding = 0) readonly buffer NodeBuffer {
  GPUNode nodes[];
};

layout(std430, binding = 1) readonly buffer MaterialBuffer {
  GPUMaterial materials[];
};

// ── Dispatch helper ───────────────────────────────────────────────────────────
// Extract the integer op or primType packed into a float's bit pattern.
// Matches gpu::packUint / gpu::unpackUint on the CPU.

int nodeOp(int idx) {
  return floatBitsToInt(nodes[idx].bboxMax.w);
}
int nodePrimType(int idx) {
  return floatBitsToInt(nodes[idx].bboxMin.w);
}

// ── Primitive dispatch ────────────────────────────────────────────────────────
// Calls the correct sdf* function from sdf_primitives.glsl based on primType.
// Include sdf_primitives.glsl before calling this.

SDF evalPrimitive(int idx, vec3 p) {
  mat4 invT = nodes[idx].invTransform;
  vec4 p0 = nodes[idx].params0;
  vec4 p1 = nodes[idx].params1;
  int matId = nodes[idx].materialId;
  vec3 color = matId >= 0 ? materials[matId].albedoRoughness.rgb : vec3(0.8);

  int type = nodePrimType(idx);

  switch (type) {
    case PRIM_SPHERE:
    return sdfSphere(p, invT, p0.x, color);
    case PRIM_BOX:
    return sdfBox(p, invT, p0.xyz, color);
    case PRIM_ROUND_BOX:
    return sdfRoundBox(p, invT, p0.xyz, p0.w, color);
    case PRIM_BOX_FRAME:
    return sdfBoxFrame(p, invT, p0.xyz, p0.w, color);
    case PRIM_ROUNDED_BOX_FRAME:
    return sdfRoundedBoxFrame(p, invT, p0.xyz, p0.w, p1.x, color);
    case PRIM_TORUS:
    return sdfTorus(p, invT, p0.x, p0.y, color);
    case PRIM_CAPPED_TORUS:
    return sdfCappedTorus(p, invT, p0.xy, p0.z, p0.w, color);
    case PRIM_LINK:
    return sdfLink(p, invT, p0.x, p0.y, p0.z, color);
    case PRIM_CYLINDER_INF:
    return sdfCylinder(p, invT, p0.xy, p0.z, color);
    case PRIM_CONE:
    return sdfCone(p, invT, p0.xy, p0.z, color);
    case PRIM_PLANE:
    return sdfPlane(p, invT, p0.xyz, p0.w, color);
    case PRIM_HEX_PRISM:
    return sdfHexPrism(p, invT, p0.x, p0.y, color);
    case PRIM_TRI_PRISM:
    return sdfTriPrism(p, invT, p0.x, p0.y, color);
    case PRIM_CAPSULE:
    return sdfCapsule(p, invT, p0.xyz, p1.xyz, p0.w, color);
    case PRIM_VERTICAL_CAPSULE:
    return sdfVerticalCapsule(p, invT, p0.x, p0.y, color);
    case PRIM_CAPPED_CYLINDER:
    return sdfCappedCylinder(p, invT, p0.x, p0.y, color);
    case PRIM_ROUNDED_CYLINDER:
    return sdfRoundedCylinder(p, invT, p0.x, p0.y, p0.z, color);
    case PRIM_CAPPED_CONE:
    return sdfCappedCone(p, invT, p0.x, p0.y, p0.z, color);
    case PRIM_ROUND_CONE:
    return sdfRoundCone(p, invT, p0.x, p0.y, p0.z, color);
    case PRIM_ELLIPSOID:
    return sdfEllipsoid(p, invT, p0.xyz, color);
    default:
    return SDF(1e10, vec3(1.0, 0.0, 1.0)); // magenta = unknown type
  }
}

// ── CSG evaluation ────────────────────────────────────────────────────────────
// Evaluate two child SDFs and combine them with the op stored on this node.

SDF applyOp(int idx, SDF left, SDF right) {
  int op = nodeOp(idx);
  float k = nodes[idx].smoothK;

  switch (op) {
    case OP_UNION:
    return opUnion(left, right);
    case OP_SUBTRACTION:
    return opSubtraction(left, right);
    case OP_INTERSECTION:
    return opIntersection(left, right);
    case OP_SMOOTH_UNION:
    return opSmoothUnion(left, right, k);
    case OP_SMOOTH_SUBTRACTION:
    return opSmoothSubtraction(left, right, k);
    case OP_SMOOTH_INTERSECTION:
    return opSmoothIntersection(left, right, k);
    default:
    return left;
  }
}
