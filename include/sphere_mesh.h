#pragma once

#include <glad/glad.h>

struct SphereMesh {
    GLuint vao, vbo, ebo;
    GLsizei indexCount;
};

// Generates a UV sphere with positions and normals, uploaded to the GPU.
// Draws with GL_TRIANGLES via an index buffer (EBO).
SphereMesh createSphere(int stacks = 32, int slices = 32);

// Frees GPU resources associated with a SphereMesh.
void destroySphere(SphereMesh& mesh);
