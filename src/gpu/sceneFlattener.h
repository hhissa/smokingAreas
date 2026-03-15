#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../scene/node.h"
#include "../scene/primitives.h"
#include "../scene/sceneGraph.h"
#include "./bvhBuilder.h"
#include "./gpuTypes.h"

// ─────────────────────────────────────────────────────────────────────────────
// SceneFlattener.h  —  Layer 3
//
// Converts a SceneGraph into the flat GPUNode array the shader reads from.
//
// Steps:
//   1. For each SceneNode in the graph that has geometry, perform a post-order
//      DFS of its CSG tree, emitting one GPUNode per CSGNode.
//      Post-order means children are emitted before parents, so a parent's
//      leftChild/rightChild indices always point to already-written entries.
//
//   2. Record the [start, end) index range for each named SceneNode in
//      nodeRanges_.  The DirtyTracker uses this to re-upload only changed
//      ranges.
//
//   3. For each SceneNode, register a BVHBuilder::Entry (world AABB + root
//   index).
//
//   4. Run BVHBuilder over all entries.  It appends BvhBranch nodes into the
//      same vector and returns the BVH root index.
//
//   5. Convert the flat Material list into GPUMaterial entries.
//
// Usage:
//   SceneFlattener f;
//   f.flatten(graph, materials);
//   // hand f.sceneBuffer() to SSBOManager
//
// No Vulkan.  No rendering.  Pure CPU data transformation.
// ─────────────────────────────────────────────────────────────────────────────

namespace gpu {

struct NodeRange {
  int32_t start; // inclusive
  int32_t end;   // exclusive
};

class SceneFlattener {
public:
  // Flatten the entire scene graph.
  // materials must be the same MaterialRegistry used when building the CSG
  // trees.
  void flatten(const scene::SceneGraph &graph,
               const scene::MaterialRegistry &materials);

  // Access the results after flatten().
  const GPUSceneBuffer &sceneBuffer() const { return buffer_; }
  const std::vector<GPUNode> &nodes() const { return nodes_; }
  const std::vector<GPUMaterial> &gpuMaterials() const { return gpuMaterials_; }

  // [start, end) index range in nodes[] for a named SceneNode.
  // Throws if the name was not present during the last flatten().
  NodeRange nodeRange(const std::string &name) const {
    auto it = nodeRanges_.find(name);
    if (it == nodeRanges_.end())
      throw std::out_of_range("SceneFlattener: unknown node '" + name + "'");
    return it->second;
  }

  bool hasNodeRange(const std::string &name) const {
    return nodeRanges_.count(name) > 0;
  }

  int32_t bvhRoot() const { return bvhRoot_; }

private:
  std::vector<GPUNode> nodes_;
  std::vector<GPUMaterial> gpuMaterials_;
  std::unordered_map<std::string, NodeRange> nodeRanges_;
  int32_t bvhRoot_ = -1;
  GPUSceneBuffer buffer_{};

  // Recursively emit GPUNodes for a CSG subtree.
  // Returns the index of the root node that was written.
  int32_t emitCsgNode(const scene::CSGNode::Ptr &node,
                      const glm::mat4 &parentWorldInv);

  // Pack primitive params into params0/params1.
  static void packParams(GPUNode &out, const scene::PrimParams &prim);

  // Convert CPU Material → GPUMaterial.
  static GPUMaterial convertMaterial(const scene::Material &m);
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

inline void SceneFlattener::flatten(const scene::SceneGraph &graph,
                                    const scene::MaterialRegistry &materials) {
  nodes_.clear();
  gpuMaterials_.clear();
  nodeRanges_.clear();
  bvhRoot_ = -1;

  // ── 1. Convert materials ──────────────────────────────────────────────────
  for (const auto &m : materials.materials())
    gpuMaterials_.push_back(convertMaterial(m));

  // ── 2. Flatten each scene node's CSG tree ─────────────────────────────────
  std::vector<BVHBuilder::Entry> bvhEntries;

  graph.traverseGeometry([&](scene::SceneNode &sceneNode) {
    int32_t rangeStart = static_cast<int32_t>(nodes_.size());

    // Pre-invert the world transform once here on the CPU.
    // Every GPUNode in this subtree inherits this inverse so the shader
    // doesn't need to invert anything per-pixel.
    glm::mat4 worldInv = glm::inverse(sceneNode.worldTransform().toMatrix());

    int32_t rootIdx = emitCsgNode(sceneNode.effectiveCsgRoot(), worldInv);

    int32_t rangeEnd = static_cast<int32_t>(nodes_.size());

    if (!sceneNode.name.empty())
      nodeRanges_[sceneNode.name] = {rangeStart, rangeEnd};

    // Register with the BVH builder using the world-space AABB.
    scene::AABB worldBBox = sceneNode.effectiveCsgRoot()->bbox;
    bvhEntries.push_back({worldBBox, rootIdx});
  });

  // ── 3. Build BVH over all scene nodes ─────────────────────────────────────
  BVHBuilder bvh;
  bvhRoot_ = bvh.build(nodes_, std::move(bvhEntries));

  // ── 4. Assemble the buffer descriptor ─────────────────────────────────────
  buffer_.nodes = nodes_.data();
  buffer_.nodeCount = static_cast<uint32_t>(nodes_.size());
  buffer_.materials = gpuMaterials_.data();
  buffer_.materialCount = static_cast<uint32_t>(gpuMaterials_.size());
  buffer_.bvhRootIndex = bvhRoot_;
}

inline int32_t SceneFlattener::emitCsgNode(const scene::CSGNode::Ptr &node,
                                           const glm::mat4 &parentWorldInv) {
  // ── Post-order: emit children first ───────────────────────────────────────
  int32_t leftIdx = -1;
  int32_t rightIdx = -1;

  if (!node->isLeaf()) {
    if (node->left)
      leftIdx = emitCsgNode(node->left, parentWorldInv);
    if (node->right)
      rightIdx = emitCsgNode(node->right, parentWorldInv);
  }

  // ── Build the GPUNode for this node ───────────────────────────────────────
  GPUNode g{};

  // bbox — pack primType into bboxMin.w and op into bboxMax.w
  g.bboxMin =
      glm::vec4(node->bbox.min,
                packUint(static_cast<uint32_t>(scene::primTypeOf(node->prim))));
  g.bboxMax =
      glm::vec4(node->bbox.max, packUint(static_cast<uint32_t>(node->op)));

  g.leftChild = leftIdx;
  g.rightChild = rightIdx;
  g.materialId = node->material.materialId;
  g.smoothK = node->smoothK;

  // ── Transform ─────────────────────────────────────────────────────────────
  // For leaf nodes, compose the scene-node world-inverse with this node's
  // own local transform, then invert to get the combined invTransform the
  // shader will use to bring world-space march points into local prim space.
  //
  // For branch nodes the transform is identity — only leaves have geometry.
  if (node->isLeaf()) {
    glm::mat4 localTRS = node->transform.toMatrix();
    glm::mat4 worldTRS = glm::inverse(parentWorldInv) * localTRS;
    g.invTransform = glm::inverse(worldTRS);
  } else {
    g.invTransform = glm::mat4(1.0f);
  }

  // ── Primitive params ──────────────────────────────────────────────────────
  if (node->isLeaf())
    packParams(g, node->prim);

  int32_t idx = static_cast<int32_t>(nodes_.size());
  nodes_.push_back(g);
  return idx;
}

inline void SceneFlattener::packParams(GPUNode &out,
                                       const scene::PrimParams &prim) {
  std::visit(
      [&](const auto &p) {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, scene::SphereParams>) {
          out.params0 = {p.radius, 0, 0, 0};
        } else if constexpr (std::is_same_v<T, scene::BoxParams>) {
          out.params0 = {p.halfExtents.x, p.halfExtents.y, p.halfExtents.z, 0};
        } else if constexpr (std::is_same_v<T, scene::RoundBoxParams>) {
          out.params0 = {p.halfExtents.x, p.halfExtents.y, p.halfExtents.z,
                         p.radius};
        } else if constexpr (std::is_same_v<T, scene::BoxFrameParams>) {
          out.params0 = {p.halfExtents.x, p.halfExtents.y, p.halfExtents.z,
                         p.edgeWidth};
        } else if constexpr (std::is_same_v<T, scene::RoundedBoxFrameParams>) {
          out.params0 = {p.halfExtents.x, p.halfExtents.y, p.halfExtents.z,
                         p.edgeWidth};
          out.params1 = {p.radius, 0, 0, 0};
        } else if constexpr (std::is_same_v<T, scene::TorusParams>) {
          out.params0 = {p.majorRadius, p.minorRadius, 0, 0};
        } else if constexpr (std::is_same_v<T, scene::CappedTorusParams>) {
          out.params0 = {p.arc.x, p.arc.y, p.outerR, p.innerR};
        } else if constexpr (std::is_same_v<T, scene::LinkParams>) {
          out.params0 = {p.halfLength, p.outerR, p.innerR, 0};
        } else if constexpr (std::is_same_v<T, scene::CylinderInfiniteParams>) {
          out.params0 = {p.centre.x, p.centre.y, p.radius, 0};
        } else if constexpr (std::is_same_v<T, scene::ConeParams>) {
          out.params0 = {p.sinCos.x, p.sinCos.y, p.height, 0};
        } else if constexpr (std::is_same_v<T, scene::PlaneParams>) {
          out.params0 = {p.normal.x, p.normal.y, p.normal.z, p.offset};
        } else if constexpr (std::is_same_v<T, scene::HexPrismParams>) {
          out.params0 = {p.radius, p.height, 0, 0};
        } else if constexpr (std::is_same_v<T, scene::TriPrismParams>) {
          out.params0 = {p.radius, p.height, 0, 0};
        } else if constexpr (std::is_same_v<T, scene::CapsuleParams>) {
          // pointA in p0.xyz, radius in p0.w, pointB in p1.xyz
          out.params0 = {p.pointA.x, p.pointA.y, p.pointA.z, p.radius};
          out.params1 = {p.pointB.x, p.pointB.y, p.pointB.z, 0};
        } else if constexpr (std::is_same_v<T, scene::VerticalCapsuleParams>) {
          out.params0 = {p.height, p.radius, 0, 0};
        } else if constexpr (std::is_same_v<T, scene::CappedCylinderParams>) {
          out.params0 = {p.radius, p.height, 0, 0};
        } else if constexpr (std::is_same_v<T, scene::RoundedCylinderParams>) {
          out.params0 = {p.bodyRadius, p.roundRadius, p.height, 0};
        } else if constexpr (std::is_same_v<T, scene::CappedConeParams>) {
          out.params0 = {p.height, p.bottomRadius, p.topRadius, 0};
        } else if constexpr (std::is_same_v<T, scene::RoundConeParams>) {
          out.params0 = {p.bottomRadius, p.topRadius, p.height, 0};
        } else if constexpr (std::is_same_v<T, scene::EllipsoidParams>) {
          out.params0 = {p.radii.x, p.radii.y, p.radii.z, 0};
        }
      },
      prim);
}

inline GPUMaterial SceneFlattener::convertMaterial(const scene::Material &m) {
  GPUMaterial g{};
  g.albedoRoughness = {m.albedo.r, m.albedo.g, m.albedo.b, m.roughness};
  g.metallicEmissive = {m.metallic, m.emissive, m.transparency, m.ior};
  return g;
}

} // namespace gpu
