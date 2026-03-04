# Compiler settings
CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

# Shader compiler
GLSLC = glslc

# Source files
SOURCES = main.cpp TextRenderer.cpp ImageFlasher.cpp
TARGET = VulkanTest

# Shader files
SHADER_SOURCES = shader.vert shader.frag text_vert.glsl text_frag.glsl image_flash.vert image_flash.frag
SHADERS = vert.spv frag.spv text_vert.spv text_frag.spv image_flash.vert.spv image_flash.frag.spv

# Default target - build everything
all: $(SHADERS) $(TARGET)

# Build executable
$(TARGET): $(SOURCES) TextRenderer.h ImageFlasher.h stb_truetype.h stb_image.h
	g++ $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Compile shaders
vert.spv: shader.vert
	$(GLSLC) shader.vert -o vert.spv

frag.spv: shader.frag
	$(GLSLC) shader.frag -o frag.spv

text_vert.spv: text_vert.glsl
	$(GLSLC) -fshader-stage=vertex text_vert.glsl -o text_vert.spv

text_frag.spv: text_frag.glsl
	$(GLSLC) -fshader-stage=fragment text_frag.glsl -o text_frag.spv

image_flash.vert.spv: image_flash.vert
	$(GLSLC) image_flash.vert -o image_flash.vert.spv

image_flash.frag.spv: image_flash.frag
	$(GLSLC) image_flash.frag -o image_flash.frag.spv

# Phony targets
.PHONY: all test clean shaders run help rebuild

# Build and run
test: all
	./$(TARGET)

run: test

# Compile only shaders
shaders: $(SHADERS)

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -f *.spv

# Rebuild everything
rebuild: clean all

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Build shaders and executable"
	@echo "  make test     - Build and run the application"
	@echo "  make run      - Same as 'make test'"
	@echo "  make shaders  - Compile only shaders"
	@echo "  make clean    - Remove executable and compiled shaders"
	@echo "  make rebuild  - Clean and rebuild everything"
	@echo "  make help     - Show this help message"
