#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "../scene/node.h"
#include "./gpuTypes.h"
// ─────────────────────────────────────────────────────────────────────────────
// BVHBuilder.h  —  Layer 3
//
// Builds a surface-area-heuristic (SAH) BVH over the scene's geometry entries
// and appends BVH interior nodes into the same flat GPUNode array that the
// CSG subtrees occupy.
//
// Key design:
//   Input:  nodes[]  — flat array already populated by SceneFlattener
//           entries  — one per SceneNode, each holding the index of that
//                      SceneNode's CSG root within nodes[] and its world AABB
//   Output: additional BVH interior GPUNodes appended to nodes[]
//           returns the index of the BVH root node
//
// The resulting flat array layout:
//   [CSG subtree for scene node 0]
//   [CSG subtree for scene node 1]
//   ...
//   [BVH interior node]
//   [BVH interior node]
//   ...
//   [BVH root node]   ← returned index
//
// The GPU raymarch shader starts at the BVH root, does AABB intersection tests
// on BvhBranch nodes (op == 7), and only descends into subtrees whose bboxes
// are hit.  When it reaches a non-BvhBranch node it starts CSG evaluation.
//
// SAH cost model:
//   cost(split) = traversalCost
//               + leftSurfaceArea/parentSurfaceArea * leftCount * intersectCost
//               + rightSurfaceArea/parentSurfaceArea * rightCount * intersectCost
//
// We evaluate all axis-aligned splits at object centroids (binned SAH with
// kBins bins per axis), which gives near-optimal quality for moderate scene
// sizes without a full sweep.
// ─────────────────────────────────────────────────────────────────────────────
 
namespace gpu {
 
class BVHBuilder {
public:
    // One entry per SceneNode handed to the BVH.
    struct Entry {
        scene::AABB aabb;        // world-space bounding box
        int32_t     nodeIndex;   // index of the CSG root GPUNode in the flat array
    };
 
    static constexpr int kBins          = 16; // SAH bins per axis
    static constexpr float kTraversalCost  = 1.0f;
    static constexpr float kIntersectCost  = 2.0f; // CSG eval is more expensive than bbox test
 
    // Build the BVH.  Appends BvhBranch GPUNodes into `nodes` and returns
    // the index of the root node.
    // Returns -1 if entries is empty.
    int32_t build(std::vector<GPUNode>& nodes,
                  std::vector<Entry>    entries);
 
private:
    std::vector<GPUNode>* nodes_ = nullptr;
 
    // Recursive builder — returns the index of the node it created.
    int32_t buildRecursive(std::vector<Entry>& entries,
                            int begin, int end);
 
    // Emit a BVH leaf pointing directly to a CSG root entry.
    int32_t emitLeaf(const Entry& entry);
 
    // Emit an interior BvhBranch node.
    int32_t emitBranch(const scene::AABB& bbox,
                        int32_t leftIdx, int32_t rightIdx);
 
    // SAH: find the best axis and split position.
    // Returns the partition index into entries[begin..end).
    // Returns -1 if no beneficial split exists (make a leaf instead).
    int partition(std::vector<Entry>& entries, int begin, int end,
                   const scene::AABB& centroidBounds);
 
    static scene::AABB computeBounds(const std::vector<Entry>& entries,
                                      int begin, int end);
    static scene::AABB computeCentroidBounds(const std::vector<Entry>& entries,
                                              int begin, int end);
    static float surfaceArea(const scene::AABB& b);
};
 
// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────
 
inline int32_t BVHBuilder::build(std::vector<GPUNode>& nodes,
                                   std::vector<Entry>    entries)
{
    if (entries.empty()) return -1;
    nodes_ = &nodes;
    return buildRecursive(entries, 0, static_cast<int>(entries.size()));
}
 
inline int32_t BVHBuilder::buildRecursive(std::vector<Entry>& entries,
                                            int begin, int end)
{
    int count = end - begin;
 
    // Base case: single entry → leaf pointing at the CSG root.
    if (count == 1) return emitLeaf(entries[begin]);
 
    scene::AABB bounds         = computeBounds(entries, begin, end);
    scene::AABB centroidBounds = computeCentroidBounds(entries, begin, end);
 
    // If all centroids are coincident, we can't split — force a two-way leaf.
    if (count == 2) {
        int32_t l = emitLeaf(entries[begin]);
        int32_t r = emitLeaf(entries[begin + 1]);
        return emitBranch(bounds, l, r);
    }
 
    int mid = partition(entries, begin, end, centroidBounds);
 
    // SAH returned no beneficial split — treat the whole group as leaves
    // under one interior node split at the midpoint.
    if (mid == -1) mid = begin + count / 2;
 
    int32_t leftIdx  = buildRecursive(entries, begin, mid);
    int32_t rightIdx = buildRecursive(entries, mid,   end);
    return emitBranch(bounds, leftIdx, rightIdx);
}
 
inline int32_t BVHBuilder::emitLeaf(const Entry& entry) {
    // A BVH leaf is just a pointer back to the existing CSG root GPUNode.
    // We don't duplicate the node; we return its existing index directly so
    // the GPU continues CSG evaluation from that index.
    return entry.nodeIndex;
}
 
inline int32_t BVHBuilder::emitBranch(const scene::AABB& bbox,
                                       int32_t leftIdx, int32_t rightIdx)
{
    GPUNode n{};
    n.bboxMin     = glm::vec4(bbox.min, packUint(static_cast<uint32_t>(scene::PrimType::Sphere))); // unused for BVH
    n.bboxMax     = glm::vec4(bbox.max, packUint(static_cast<uint32_t>(scene::CSGOp::BvhBranch)));
    n.leftChild   = leftIdx;
    n.rightChild  = rightIdx;
    n.materialId  = -1;
    n.smoothK     = 0.0f;
    n.invTransform= glm::mat4(1.0f); // identity — BVH nodes have no transform
 
    int32_t idx = static_cast<int32_t>(nodes_->size());
    nodes_->push_back(n);
    return idx;
}
 
inline int BVHBuilder::partition(std::vector<Entry>& entries,
                                   int begin, int end,
                                   const scene::AABB& centroidBounds)
{
    // Binned SAH — evaluate kBins splits on each axis, pick the best.
    float   bestCost  = std::numeric_limits<float>::max();
    int     bestAxis  = -1;
    int     bestBin   = -1;
    int     count     = end - begin;
    float   parentSA  = surfaceArea(computeBounds(entries, begin, end));
 
    if (parentSA < 1e-12f) return -1;
 
    for (int axis = 0; axis < 3; ++axis) {
        float lo = centroidBounds.min[axis];
        float hi = centroidBounds.max[axis];
        if (hi - lo < 1e-6f) continue; // degenerate on this axis
 
        // ── Bin the entries ───────────────────────────────────────────────
        struct Bin { scene::AABB bounds; int count = 0; };
        Bin bins[kBins];
 
        float scale = static_cast<float>(kBins) / (hi - lo);
        for (int i = begin; i < end; ++i) {
            glm::vec3 c = entries[i].aabb.centre();
            float     f = (c[axis] - lo) * scale;
            // Guard: clamp before cast — inf or nan from oversized AABBs
            // would invoke UB and fire SIGFPE on x86.
            if (!std::isfinite(f)) f = 0.0f;
            int b = std::clamp(static_cast<int>(f), 0, kBins - 1);
            bins[b].count++;
            if (bins[b].count == 1)
                bins[b].bounds = entries[i].aabb;
            else
                bins[b].bounds = bins[b].bounds.merge(entries[i].aabb);
        }
 
        // ── Sweep left → right, accumulate prefix SA and count ────────────
        float leftSA[kBins - 1];
        int   leftCount[kBins - 1];
        {
            scene::AABB acc;
            int cnt = 0;
            bool started = false;
            for (int b = 0; b < kBins - 1; ++b) {
                if (bins[b].count > 0) {
                    acc     = started ? acc.merge(bins[b].bounds) : bins[b].bounds;
                    started = true;
                }
                cnt      += bins[b].count;
                leftSA[b]    = started ? surfaceArea(acc) : 0.0f;
                leftCount[b] = cnt;
            }
        }
 
        // ── Sweep right → left, evaluate SAH cost for each split ──────────
        {
            scene::AABB acc;
            int cnt = 0;
            bool started = false;
            for (int b = kBins - 1; b >= 1; --b) {
                if (bins[b].count > 0) {
                    acc     = started ? acc.merge(bins[b].bounds) : bins[b].bounds;
                    started = true;
                }
                cnt += bins[b].count;
 
                int   lc   = leftCount[b - 1];
                float lsa  = leftSA[b - 1];
                float rsa  = started ? surfaceArea(acc) : 0.0f;
                int   rc   = cnt;
 
                if (lc == 0 || rc == 0) continue;
 
                float cost = kTraversalCost
                           + (lsa / parentSA) * lc * kIntersectCost
                           + (rsa / parentSA) * rc * kIntersectCost;
 
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestBin  = b;
                }
            }
        }
    }
 
    // If no split beats the leaf cost, signal the caller to make a leaf.
    float leafCost = count * kIntersectCost;
    if (bestAxis == -1 || bestCost >= leafCost) return -1;
 
    // ── Partition entries around the best split ───────────────────────────
    float lo    = centroidBounds.min[bestAxis];
    float hi    = centroidBounds.max[bestAxis];
    float scale = kBins / (hi - lo);
 
    auto it = std::partition(entries.begin() + begin, entries.begin() + end,
        [&](const Entry& e) {
            int b = static_cast<int>((e.aabb.centre()[bestAxis] - lo) * scale);
            b = std::clamp(b, 0, kBins - 1);
            return b < bestBin;
        });
 
    int mid = static_cast<int>(it - entries.begin());
 
    // Guard: partition must not produce an empty side.
    if (mid == begin || mid == end) return begin + count / 2;
    return mid;
}
 
inline scene::AABB BVHBuilder::computeBounds(const std::vector<Entry>& entries,
                                              int begin, int end)
{
    scene::AABB b;
    bool started = false;
    for (int i = begin; i < end; ++i) {
        b = started ? b.merge(entries[i].aabb) : entries[i].aabb;
        started = true;
    }
    return b;
}
 
inline scene::AABB BVHBuilder::computeCentroidBounds(const std::vector<Entry>& entries,
                                                       int begin, int end)
{
    scene::AABB b;
    bool started = false;
    for (int i = begin; i < end; ++i) {
        glm::vec3 c = entries[i].aabb.centre();
        scene::AABB pt{ c, c };
        b = started ? b.merge(pt) : pt;
        started = true;
    }
    return b;
}
 
inline float BVHBuilder::surfaceArea(const scene::AABB& b) {
    glm::vec3 e = b.extent();
    return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
}
 
} // namespace gpu
 // namespace gpu
