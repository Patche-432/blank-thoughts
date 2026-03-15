#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

Camera::Camera(glm::vec3 startPos, float startYaw, float startPitch)
    : position(startPos), yaw(startYaw), pitch(startPitch)
{
    updateVectors();
}

void Camera::processKeyboard(bool w, bool a, bool s, bool d, float dt) {
    float velocity = moveSpeed * dt;
    if (w) position += front * velocity;
    if (s) position -= front * velocity;
    if (a) position -= right * velocity;
    if (d) position += right * velocity;
}

void Camera::processMouseMovement(float xOffset, float yOffset) {
    yaw   += xOffset * mouseSensitivity;
    pitch += yOffset * mouseSensitivity;

    // Clamp pitch so we don't flip over
    pitch = std::clamp(pitch, -89.0f, 89.0f);

    updateVectors();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + front, up);
}

glm::vec3 Camera::getFront() const {
    return front;
}

void Camera::updateVectors() {
    glm::vec3 f;
    f.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    f.y = sin(glm::radians(pitch));
    f.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(f);
    right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    up    = glm::normalize(glm::cross(right, front));
}
