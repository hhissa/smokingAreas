#include "./sceneValidator.h"

#include <iostream>
#include <unordered_set>
#include <unordered_map>

namespace scene {

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

bool SceneValidator::validate(const SceneGraph& graph) {
    issues_.clear();
    checkSceneNodes(graph);

    // Validate every CSG tree in the scene.
    graph.traverseGeometry([this](SceneNode& node) {
        checkCsgTree(node.name, node.effectiveCsgRoot(), 0);
    });

    return !hasErrors();
}

bool SceneValidator::hasErrors() const {
    for (const auto& i : issues_)
        if (i.isError()) return true;
    return false;
}

bool SceneValidator::hasWarnings() const {
    for (const auto& i : issues_)
        if (i.isWarning()) return true;
    return false;
}

void SceneValidator::report() const {
    for (const auto& issue : issues_) {
        const char* prefix = issue.isError() ? "[ERROR]" : "[WARN] ";
        std::cerr << prefix << " [" << issue.nodeName << "] "
                  << issue.message << "\n";
    }
    if (issues_.empty()) {
        std::cerr << "[SceneValidator] No issues found.\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene-level checks
// ─────────────────────────────────────────────────────────────────────────────

void SceneValidator::checkSceneNodes(const SceneGraph& graph) {
    // Track names to detect duplicates and instance targets to detect cycles.
    std::unordered_map<std::string, int> nameCounts;
    std::unordered_set<SceneNode*>       visiting; // for cycle detection

    // First pass: count names across the entire graph.
    graph.traverse([&](SceneNode& node) {
        if (!node.name.empty() && node.name != "__root__")
            nameCounts[node.name]++;
    });

    // Report duplicate names.
    for (const auto& [name, count] : nameCounts) {
        if (count > 1) {
            addError(name, "Duplicate node name appears " +
                           std::to_string(count) + " times. "
                           "SceneGraph::find() will return the first match.");
        }
    }

    // Second pass: per-node checks.
    graph.traverse([&](SceneNode& node) {
        if (node.name == "__root__") return;

        // Warn about orphan group nodes (no geometry, no children).
        if (!node.hasGeometry() && node.children.empty()) {
            addWarning(node.name, "Node has no geometry and no children.");
        }

        // Instance cycle detection using DFS.
        if (node.isInstance()) {
            visiting.clear();
            visiting.insert(&node);

            SceneNode* cursor = node.instanceOf.lock().get();
            while (cursor) {
                if (visiting.count(cursor)) {
                    addError(node.name, "Instance cycle detected.");
                    break;
                }
                visiting.insert(cursor);
                auto src = cursor->instanceOf.lock();
                cursor = src ? src.get() : nullptr;
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// CSG tree checks (recursive DFS)
// ─────────────────────────────────────────────────────────────────────────────

void SceneValidator::checkCsgTree(const std::string& ownerName,
                                   const CSGNode::Ptr& node,
                                   int depth)
{
    if (!node) return;

    const std::string label = ownerName + "/" +
                              (node->name.empty() ? "<anon>" : node->name);

    // ── Depth limit ───────────────────────────────────────────────────────────
    if (depth > kMaxCsgDepth) {
        addError(label, "CSG tree depth exceeds kMaxCsgDepth (" +
                        std::to_string(kMaxCsgDepth) + "). "
                        "The GPU stack will overflow.");
        return; // Don't recurse further — error already reported.
    }

    if (node->isLeaf()) {
        // ── Degenerate AABB ───────────────────────────────────────────────────
        const AABB& bb = node->bbox;
        if (bb.min.x > bb.max.x ||
            bb.min.y > bb.max.y ||
            bb.min.z > bb.max.z)
        {
            addError(label, "Degenerate AABB (min > max). "
                            "Was updateAABB() called after construction?");
        }

        // ── Missing material ──────────────────────────────────────────────────
        if (node->material.materialId < 0) {
            addWarning(label, "Leaf node has no registered material "
                              "(materialId == -1). Will render with defaults.");
        }

    } else {
        // ── Null children ─────────────────────────────────────────────────────
        if (!node->left) {
            addError(label, "Branch node has null left child.");
        }
        if (!node->right) {
            addError(label, "Branch node has null right child.");
        }

        // ── Invalid smoothK ───────────────────────────────────────────────────
        if ((node->op == CSGOp::SmoothUnion        ||
             node->op == CSGOp::SmoothSubtraction   ||
             node->op == CSGOp::SmoothIntersection) &&
             node->smoothK < 0.0f)
        {
            addError(label, "Smooth operation has negative smoothK (" +
                            std::to_string(node->smoothK) + ").");
        }

        // ── AABB sanity ───────────────────────────────────────────────────────
        const AABB& bb = node->bbox;
        if (bb.min.x > bb.max.x ||
            bb.min.y > bb.max.y ||
            bb.min.z > bb.max.z)
        {
            addWarning(label, "Branch node has degenerate AABB. "
                              "BVH culling may discard this subtree incorrectly.");
        }

        // ── Recurse ───────────────────────────────────────────────────────────
        checkCsgTree(ownerName, node->left,  depth + 1);
        checkCsgTree(ownerName, node->right, depth + 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue helpers
// ─────────────────────────────────────────────────────────────────────────────

void SceneValidator::addError(const std::string& node, const std::string& msg) {
    issues_.push_back({ ValidationIssue::Severity::Error, node, msg });
}

void SceneValidator::addWarning(const std::string& node, const std::string& msg) {
    issues_.push_back({ ValidationIssue::Severity::Warning, node, msg });
}

} // namespace scene
