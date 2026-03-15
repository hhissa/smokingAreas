#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "./sceneNode.h"

// ─────────────────────────────────────────────────────────────────────────────
// SceneGraph.h  —  Layer 2
//
// Owns the root SceneNode and provides the top-level scene manipulation API.
//
// Responsibilities:
//   addNode()   — attach a node under an existing parent (or the root)
//   find()      — deep name search, returns nullptr if not found
//   instance()  — create a new node that re-uses an existing node's CSG tree
//   clear()     — reset to an empty scene
//
// The SceneGraph does NOT:
//   - Evaluate animations      (AnimationSystem does that)
//   - Upload anything to the GPU (DirtyTracker / SSBOManager do that)
//   - Know about flattening     (SceneFlattener does that)
//
// Thread safety: none.  Mutate from the main thread only.
// ─────────────────────────────────────────────────────────────────────────────

namespace scene {

class SceneGraph {
public:
  SceneGraph() { root_ = SceneNode::make("__root__"); }

  // ── Root access ───────────────────────────────────────────────────────────
  SceneNode::Ptr root() const { return root_; }

  // ── Add a node ────────────────────────────────────────────────────────────
  // Attaches `node` as a child of `parent`.
  // If parent is nullptr, attaches directly under the scene root.
  // Returns `node` so calls can be chained.
  SceneNode::Ptr addNode(SceneNode::Ptr node, SceneNode::Ptr parent = nullptr) {
    if (!parent)
      parent = root_;
    parent->addChild(node);
    return node;
  }

  // Convenience: create a named node and attach it in one call.
  SceneNode::Ptr addNode(const std::string &name,
                         SceneNode::Ptr parent = nullptr) {
    return addNode(SceneNode::make(name), std::move(parent));
  }

  // ── Find by name ──────────────────────────────────────────────────────────
  // Depth-first search from the root.
  // Returns nullptr if no node with that name exists.
  SceneNode::Ptr find(const std::string &name) const {
    return findRecursive(root_, name);
  }

  // find() variant that throws if the node is not found.
  SceneNode::Ptr require(const std::string &name) const {
    auto n = find(name);
    if (!n)
      throw std::runtime_error("SceneGraph::require: node '" + name +
                               "' not found.");
    return n;
  }

  // ── Instancing ────────────────────────────────────────────────────────────
  // Creates a new node that shares the CSG tree of `source`.
  // The instance gets its own world transform via `offsetTransform`.
  // Parent defaults to the scene root.
  //
  // Example: four identical table legs placed at different positions.
  //   auto leg = graph.require("leg_prototype");
  //   for (auto& pos : legPositions) {
  //       graph.instance(leg, Transform::translate(pos), legsGroup);
  //   }
  SceneNode::Ptr instance(SceneNode::Ptr source,
                          const Transform &offsetTransform,
                          SceneNode::Ptr parent = nullptr) {
    static int instanceCounter = 0;
    auto inst = SceneNode::make(source->name + "_inst" +
                                std::to_string(instanceCounter++));
    inst->localTransform = offsetTransform;
    inst->instanceOf = source;
    return addNode(inst, std::move(parent));
  }

  // ── Scene reset ───────────────────────────────────────────────────────────
  void clear() { root_ = SceneNode::make("__root__"); }

  // ── Traversal ─────────────────────────────────────────────────────────────
  // Depth-first visit of every node in the scene.
  // Callback signature: void(SceneNode&)
  template <typename Fn> void traverse(Fn &&fn) const {
    traverseRecursive(root_, fn);
  }

  // Visit only nodes that have geometry (own or instanced csgRoot).
  template <typename Fn> void traverseGeometry(Fn &&fn) const {
    traverseRecursive(root_, [&](SceneNode &node) {
      if (node.hasGeometry())
        fn(node);
    });
  }

  // Mark the entire scene dirty (e.g. after a full scene swap).
  void markAllDirty() { root_->markDirtyRecursive(); }

private:
  SceneNode::Ptr root_;

  // ── Internal helpers ──────────────────────────────────────────────────────

  static SceneNode::Ptr findRecursive(const SceneNode::Ptr &node,
                                      const std::string &name) {
    if (node->name == name)
      return node;
    for (const auto &child : node->children) {
      auto result = findRecursive(child, name);
      if (result)
        return result;
    }
    return nullptr;
  }

  template <typename Fn>
  static void traverseRecursive(const SceneNode::Ptr &node, Fn &&fn) {
    fn(*node);
    for (const auto &child : node->children) {
      traverseRecursive(child, fn);
    }
  }
};

} // namespace scene
