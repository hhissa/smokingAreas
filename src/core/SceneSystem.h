#pragma once

#include <functional>
#include <string>
#include <vector>
#include <stdexcept>

#include "../scene/sceneGraph.h"
#include "../scene/material.h"
#include "../scene/sceneValidator.h"

// ─────────────────────────────────────────────────────────────────────────────
// SceneSystem
//
// Holds a list of named scene-build functions.
// Calling load() clears the scene graph and material registry, runs the
// builder for the current scene, validates, then sets the dirty flag.
//
// Each builder is a free function or lambda with the signature:
//   void(scene::SceneGraph&, scene::MaterialRegistry&)
//
// Application registers one builder per scene and calls
// sceneSystem_.load(sceneGraph_, materials_) whenever the scene changes.
// ─────────────────────────────────────────────────────────────────────────────

using SceneBuilder = std::function<void(scene::SceneGraph&, scene::MaterialRegistry&)>;

struct SceneEntry {
    std::string  name;
    SceneBuilder build;
};

class SceneSystem {
public:
    void add(const std::string& name, SceneBuilder builder) {
        entries_.push_back({ name, std::move(builder) });
    }

    // ── Navigation ────────────────────────────────────────────────────────────
    void next() { if (!entries_.empty()) index_ = (index_ + 1) % entries_.size(); }
    void prev() { if (!entries_.empty()) index_ = (index_ + entries_.size() - 1) % entries_.size(); }
    void setIndex(int i) { index_ = static_cast<size_t>(i) % entries_.size(); }

    int         currentIndex() const { return static_cast<int>(index_); }
    int         count()        const { return static_cast<int>(entries_.size()); }
    const std::string& currentName() const { return entries_[index_].name; }

    // ── Load ──────────────────────────────────────────────────────────────────
    // Clears graph + materials, runs the active builder, validates.
    // Returns false and prints to stderr if validation fails (scene not loaded).
    bool load(scene::SceneGraph& graph, scene::MaterialRegistry& materials) const {
        if (entries_.empty()) throw std::runtime_error("SceneSystem: no scenes registered.");

        graph.clear();
        materials.clear();

        entries_[index_].build(graph, materials);

        scene::SceneValidator validator;
        if (!validator.validate(graph)) {
            validator.report();
            std::fprintf(stderr, "[SceneSystem] Scene '%s' failed validation.\n",
                         entries_[index_].name.c_str());
            return false;
        }
        std::fprintf(stderr, "[SceneSystem] Loaded scene '%s' (%d).\n",
                     entries_[index_].name.c_str(), static_cast<int>(index_));
        return true;
    }

private:
    std::vector<SceneEntry> entries_;
    size_t                  index_ = 0;
};;
