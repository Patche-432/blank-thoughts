#pragma once

#include <glm/glm.hpp>

// FPS-style free-look camera.
// Call processKeyboard() each frame, processMouseMovement() from a GLFW cursor callback.
class Camera {
public:
    // Position and orientation
    glm::vec3 position;
    float yaw;    // degrees, horizontal
    float pitch;  // degrees, vertical (clamped)

    // Settings
    float moveSpeed    = 10.0f;
    float mouseSensitivity = 0.1f;
    float fov          = 60.0f;

    Camera(glm::vec3 startPos = glm::vec3(0.0f, 5.0f, 50.0f),
           float startYaw     = -90.0f,
           float startPitch   = 0.0f);

    // Call once per frame with delta time and which WASD keys are held
    void processKeyboard(bool w, bool a, bool s, bool d, float dt);

    // Call from GLFW cursor position callback with pixel offset since last frame
    void processMouseMovement(float xOffset, float yOffset);

    // Returns a view matrix for use in shaders
    glm::mat4 getViewMatrix() const;

    // Returns the normalised forward vector
    glm::vec3 getFront() const;

private:
    void updateVectors();
    glm::vec3 front;
    glm::vec3 right;
    glm::vec3 up;
};
