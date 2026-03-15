#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "./node.h"
#include "./transform.h"

// ─────────────────────────────────────────────────────────────────────────────
// SceneNode.h  —  Layer 2
//
// A node in the scene hierarchy.  Distinct from CSGNode:
//
//   CSGNode   — describes the *shape* of one object (what it looks like)
//   SceneNode — describes the *placement* of an object in the world
//               (where it is, what it's parented to, whether it's an instance)
//
// A SceneNode optionally holds a csgRoot — the root of a CSG tree that
// defines its geometry.  SceneNodes without a csgRoot are pure transform
// groups (useful for parenting a set of objects without adding geometry).
//
// Instancing:
//   When instanceOf is set, this node re-uses another node's CSG tree at a
//   different world transform.  The flattener writes the same GPUNode data
//   twice with a different invTransform each time.  No geometry is duplicated.
//
// worldTransform():
//   Walks up the parent chain composing local transforms.
//   The flattener inverts the result and stores it in GPUNode.invTransform.
//
// enable_shared_from_this:
//   Required so addChild() can store a valid weak_ptr to this node as the
//   child's parent without constructing a second owning shared_ptr from `this`.
//   Nodes MUST be created via std::make_shared — never stack-allocated.
//
// No Vulkan.  No GPU types.  No animation evaluation.
// ─────────────────────────────────────────────────────────────────────────────

namespace scene {

struct SceneNode : public std::enable_shared_from_this<SceneNode> {
  using Ptr = std::shared_ptr<SceneNode>;
  using WeakPtr = std::weak_ptr<SceneNode>;

  // ── Identity ──────────────────────────────────────────────────────────────
  std::string name;

  // ── Transform ─────────────────────────────────────────────────────────────
  Transform localTransform = Transform::identity();

  // ── Geometry ──────────────────────────────────────────────────────────────
  // Null for pure group nodes.
  CSGNode::Ptr csgRoot;

  // ── Hierarchy ─────────────────────────────────────────────────────────────
  // Weak to avoid retain cycles: parent owns children, children don't own
  // parent.
  WeakPtr parent;
  std::vector<Ptr> children;

  // ── Instancing ────────────────────────────────────────────────────────────
  // When set, this node borrows the csgRoot of the target node.
  // The node's own csgRoot is ignored by the flattener.
  WeakPtr instanceOf;

  // ── Animation ─────────────────────────────────────────────────────────────
  // Index into AnimationSystem's track list. -1 = not animated.
  int animTrackId = -1;

  // ── Dirty flag ────────────────────────────────────────────────────────────
  // Set whenever localTransform changes or a child is added/removed.
  // DirtyTracker reads this to decide what ranges need re-uploading.
  bool dirty = true;

  // ── Construction helper ───────────────────────────────────────────────────
  // Always create nodes through make() so enable_shared_from_this is active.
  static Ptr make(const std::string &name = "") {
    auto n = std::make_shared<SceneNode>();
    n->name = name;
    return n;
  }

  // ── World transform ───────────────────────────────────────────────────────
  // Walks the parent chain and composes transforms root → leaf.
  Transform worldTransform() const {
    auto p = parent.lock();
    if (!p)
      return localTransform;
    return p->worldTransform().compose(localTransform);
  }

  // ── CSG root resolution ───────────────────────────────────────────────────
  // Returns this node's own csgRoot, or the instance source's root,
  // or nullptr if neither has geometry.
  CSGNode::Ptr effectiveCsgRoot() const {
    if (csgRoot)
      return csgRoot;
    auto src = instanceOf.lock();
    if (src)
      return src->effectiveCsgRoot();
    return nullptr;
  }

  bool isInstance() const { return !instanceOf.expired(); }
  bool hasGeometry() const { return effectiveCsgRoot() != nullptr; }

  // ── Child management ──────────────────────────────────────────────────────
  // Adds `child` under this node and sets child->parent correctly.
  // The child must not already have a parent.
  void addChild(Ptr child) {
    child->parent = weak_from_this();
    children.push_back(std::move(child));
    dirty = true;
  }

  // Detaches the first child with the given name.  No-op if not found.
  void removeChild(const std::string &childName) {
    auto it = std::find_if(children.begin(), children.end(),
                           [&](const Ptr &c) { return c->name == childName; });
    if (it != children.end()) {
      (*it)->parent.reset();
      children.erase(it);
      dirty = true;
    }
  }

  // ── Dirty propagation ─────────────────────────────────────────────────────
  // Any transform change must call this so all descendants get re-uploaded.
  void markDirtyRecursive() {
    dirty = true;
    for (auto &c : children)
      c->markDirtyRecursive();
  }
};

} // namespace scene
