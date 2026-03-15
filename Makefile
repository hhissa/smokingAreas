# ─────────────────────────────────────────────────────────────────────────────
# Makefile — SDF Renderer (compute pipeline)
#
# Targets:
#   make          — build the binary  (default)
#   make shaders  — compile all GLSL shaders to SPIR-V
#   make all      — shaders + binary
#   make run      — build everything and launch
#   make clean    — remove build artefacts
#   make info     — print resolved tool paths and flags
# ─────────────────────────────────────────────────────────────────────────────

# ── Output names ─────────────────────────────────────────────────────────────

TARGET  := sdf-renderer
BUILD   := build

# ── Source files ──────────────────────────────────────────────────────────────
# Collect every .cpp in the project tree automatically.
# Add new translation units anywhere under src/ without touching this file.

SRCS := main.cpp \
        src/core/VulkanContext.cpp \
        src/core/Swapchain.cpp \
        src/core/ComputePass.cpp \
        src/core/Application.cpp \
        src/scene/sceneValidator.cpp

OBJS := $(SRCS:%.cpp=$(BUILD)/%.o)
DEPS := $(OBJS:.o=.d)

# ── Shader files ──────────────────────────────────────────────────────────────
# Every .vert / .frag / .comp under shaders/ is compiled to a .spv alongside
# the source file so the binary can find it with a relative path.

GLSL_SRCS := $(shell find shaders -name '*.vert' -o -name '*.frag' -o -name '*.comp')
SPIRV_OUTS := $(GLSL_SRCS:%=%.spv)

# ── Compiler ─────────────────────────────────────────────────────────────────

CXX      := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wpedantic \
            -MMD -MP                        \
            -I.                             \
            -Isrc                           \
            -Isrc/core                      \
            -Isrc/scene                     \
            -Isrc/gpu

# ── Platform detection ────────────────────────────────────────────────────────
# GLFW and Vulkan are found differently on macOS (Homebrew) vs Linux.

UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    # Homebrew paths — adjust prefix if you use a non-standard location.
    BREW_PREFIX ?= $(shell brew --prefix 2>/dev/null || echo /usr/local)

    CXXFLAGS += -I$(BREW_PREFIX)/include
    LDFLAGS  := -L$(BREW_PREFIX)/lib \
                -lglfw \
                -lvulkan \
                -framework Cocoa \
                -framework IOKit \
                -framework CoreVideo

    # MoltenVK ships glslc inside the Vulkan SDK; fall back to glslangValidator.
    GLSLC    := $(shell which glslc 2>/dev/null || echo glslangValidator)

else
    # Linux — assumes system packages or a Vulkan SDK on PATH.
    CXXFLAGS += $(shell pkg-config --cflags glfw3 2>/dev/null)
    LDFLAGS  := $(shell pkg-config --libs   glfw3 2>/dev/null) \
                -lvulkan

    GLSLC    := $(shell which glslc 2>/dev/null || echo glslangValidator)
endif

# glslangValidator needs a different output flag (-o vs positional argument).
GLSLC_FLAGS := $(if $(findstring glslangValidator,$(GLSLC)),-V -o,-o)
GLSLC_INCLUDES := -I shaders/include

# ── SPIR-V rules ──────────────────────────────────────────────────────────────
# Pattern rule: compile any GLSL stage to .spv next to the source file.
# The binary loads shaders by filename (e.g. "shaders/triangle.comp.spv"),
# so keeping them beside the sources keeps paths simple.

%.spv: %
	@echo "  GLSL  $<"
	$(GLSLC) $(GLSLC_INCLUDES) $(GLSLC_FLAGS) $@ $<

shaders: $(SPIRV_OUTS)

# ── C++ compilation ───────────────────────────────────────────────────────────
# Object files mirror the source tree under $(BUILD)/ so that parallel builds
# (-j) don't collide and `make clean` removes only generated files.

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ── Link ──────────────────────────────────────────────────────────────────────

$(TARGET): $(OBJS)
	@echo "  LD    $@"
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

# ── Convenience targets ───────────────────────────────────────────────────────

.PHONY: all run clean info shaders

all: shaders $(TARGET)

run: all
	./$(TARGET)

clean:
	@echo "  CLEAN"
	rm -rf $(BUILD) $(TARGET) $(SPIRV_OUTS)

info:
	@echo "CXX:      $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "LDFLAGS:  $(LDFLAGS)"
	@echo "GLSLC:    $(GLSLC)"
	@echo "SRCS:     $(SRCS)"
	@echo "GLSL:     $(GLSL_SRCS)"
	@echo "SPIRV:    $(SPIRV_OUTS)"

# ── Auto-generated dependency includes ───────────────────────────────────────
# -MMD -MP above causes g++ to emit a .d file alongside each .o.
# Including them here means editing a header triggers a rebuild of all
# translation units that include it — without any manual dependency tracking.

-include $(DEPS)

# ── Default target ────────────────────────────────────────────────────────────

.DEFAULT_GOAL := $(TARGET)
