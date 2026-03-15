#pragma once

#include <string>
#include <vector>

#include "./node.h"
#include "./sceneGraph.h"

// ─────────────────────────────────────────────────────────────────────────────
// SceneValidator.h  —  Layer 2
//
// Validates the scene graph and every CSG tree within it before flattening.
// Run once after scene construction, not every frame.
//
// Issues are collected into a vector rather than thrown immediately so the
// caller can decide whether to hard-fail or print warnings and continue.
//
// Checks performed:
//   SceneNode level:
//     - Duplicate node names (makes find() ambiguous)
//     - Instance cycles (A instances B which instances A)
//     - Nodes with no geometry and no children (orphan group nodes)
//
//   CSG tree level (per node that has geometry):
//     - Degenerate AABB (min > max in any axis)
//     - Missing material (materialId == -1) on leaf nodes
//     - Excessive tree depth (> kMaxCsgDepth — GPU stack overflow risk)
//     - Branch nodes with a null left or right child
//     - smoothK < 0 on smooth operations
// ─────────────────────────────────────────────────────────────────────────────

namespace scene {

struct ValidationIssue {
  enum class Severity { Warning, Error };

  Severity severity;
  std::string nodeName; // SceneNode name, or CSG node name if applicable
  std::string message;

  bool isError() const { return severity == Severity::Error; }
  bool isWarning() const { return severity == Severity::Warning; }
};

class SceneValidator {
public:
  // Maximum CSG tree depth before a depth-limit error is raised.
  // The GPU stack in the raymarch shader is sized to this limit.
  static constexpr int kMaxCsgDepth = 32;

  // Run all checks.  Returns true if no errors were found (warnings are ok).
  bool validate(const SceneGraph &graph);

  const std::vector<ValidationIssue> &issues() const { return issues_; }
  bool hasErrors() const;
  bool hasWarnings() const;

  // Print all issues to stderr.
  void report() const;

private:
  std::vector<ValidationIssue> issues_;

  void checkSceneNodes(const SceneGraph &graph);
  void checkCsgTree(const std::string &ownerName, const CSGNode::Ptr &node,
                    int depth);

  void addError(const std::string &node, const std::string &msg);
  void addWarning(const std::string &node, const std::string &msg);
};

} // namespace scene
