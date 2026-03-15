#include "sphere_mesh.h"

#include <vector>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

SphereMesh createSphere(int stacks, int slices) {
    std::vector<float>        verts;
    std::vector<unsigned int> indices;

    for (int i = 0; i <= stacks; i++) {
        float v   = (float)i / stacks;
        float phi = v * glm::pi<float>();

        for (int j = 0; j <= slices; j++) {
            float u     = (float)j / slices;
            float theta = u * glm::two_pi<float>();

            float x = std::sin(phi) * std::cos(theta);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(theta);

            // Position
            verts.push_back(x);
            verts.push_back(y);
            verts.push_back(z);

            // Normal (same as position on a unit sphere)
            verts.push_back(x);
            verts.push_back(y);
            verts.push_back(z);
        }
    }

    for (int i = 0; i < stacks; i++) {
        for (int j = 0; j < slices; j++) {
            unsigned int row0 = i       * (slices + 1) + j;
            unsigned int row1 = (i + 1) * (slices + 1) + j;

            indices.push_back(row0);
            indices.push_back(row1);
            indices.push_back(row0 + 1);

            indices.push_back(row1);
            indices.push_back(row1 + 1);
            indices.push_back(row0 + 1);
        }
    }

    SphereMesh mesh;
    mesh.indexCount = static_cast<GLsizei>(indices.size());

    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glGenBuffers(1, &mesh.ebo);

    glBindVertexArray(mesh.vao);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // aPos — location 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // aNormal — location 1
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    return mesh;
}

void destroySphere(SphereMesh& mesh) {
    glDeleteBuffers(1, &mesh.ebo);
    glDeleteBuffers(1, &mesh.vbo);
    glDeleteVertexArrays(1, &mesh.vao);
    mesh.ebo = mesh.vbo = mesh.vao = 0;
    mesh.indexCount = 0;
}
