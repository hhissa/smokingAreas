#version 450

// Fullscreen triangle — no vertex buffer needed.
// Produces a triangle that covers the entire screen with correct UVs.
layout(location = 0) out vec2 outUV;

void main() {
    // vertices at clip space: (-1,-1), (3,-1), (-1,3)
    vec2 pos = vec2(
        (gl_VertexIndex == 1) ? 3.0 : -1.0,
        (gl_VertexIndex == 2) ? 3.0 : -1.0
    );
    outUV = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
