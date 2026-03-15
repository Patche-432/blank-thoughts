#include <iostream>
#include <stdexcept>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "gl_utils.h"
#include "shader.h"
#include "sphere_mesh.h"
#include "camera.h"

// -------------------------
// Config
// -------------------------
static constexpr int   WINDOW_W  = 1280;
static constexpr int   WINDOW_H  = 720;
static constexpr float NEAR_Z    = 0.1f;
static constexpr float FAR_Z     = 500.0f;
static constexpr float FOV_MIN   = 10.0f;
static constexpr float FOV_MAX   = 120.0f;
static constexpr float ZOOM_STEP = 2.0f;

// -------------------------
// Globals for callbacks
// -------------------------
static Camera g_camera;
static float  g_lastMouseX = WINDOW_W / 2.0f;
static float  g_lastMouseY = WINDOW_H / 2.0f;
static bool   g_firstMouse = true;
static float  g_fov        = 60.0f;

// -------------------------
// GLFW callbacks
// -------------------------
void mouseCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    float x = static_cast<float>(xpos);
    float y = static_cast<float>(ypos);

    if (g_firstMouse) {
        g_lastMouseX = x;
        g_lastMouseY = y;
        g_firstMouse = false;
    }

    float xOffset = -(x - g_lastMouseX); // inverted left/right
    float yOffset =  (y - g_lastMouseY); // inverted up/down

    g_lastMouseX = x;
    g_lastMouseY = y;

    g_camera.processMouseMovement(xOffset, yOffset);
}

void scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    g_fov -= static_cast<float>(yoffset) * ZOOM_STEP;
    if (g_fov < FOV_MIN) g_fov = FOV_MIN;
    if (g_fov > FOV_MAX) g_fov = FOV_MAX;
}

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// -------------------------
// Uniform helpers
// -------------------------
static void setSunUniforms(GLuint prog,
                           const glm::mat4& model,
                           const glm::mat4& view,
                           const glm::mat4& proj)
{
    glUniformMatrix4fv(glGetUniformLocation(prog, "model"),      1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(prog, "view"),       1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(prog, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
}

static void setPlanetUniforms(GLuint prog,
                              const glm::mat4& model,
                              const glm::mat4& view,
                              const glm::mat4& proj,
                              const glm::vec3& color,
                              const glm::vec3& lightPos,
                              const glm::vec3& lightColor)
{
    glUniformMatrix4fv(glGetUniformLocation(prog, "model"),      1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(prog, "view"),       1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(prog, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform3fv(glGetUniformLocation(prog, "objectColor"), 1, glm::value_ptr(color));
    glUniform3fv(glGetUniformLocation(prog, "lightPos"),    1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(prog, "lightColor"),  1, glm::value_ptr(lightColor));
}

// -------------------------
// Entry point
// -------------------------
int main() {
    try {
        if (!glfwInit())
            throw std::runtime_error("Failed to initialize GLFW");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(WINDOW_W, WINDOW_H, "Solar System", nullptr, nullptr);
        if (!window)
            throw std::runtime_error("Failed to create GLFW window");

        glfwMakeContextCurrent(window);

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetKeyCallback(window, keyCallback);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
            throw std::runtime_error("Failed to initialize GLAD");

        glEnable(GL_DEPTH_TEST);

        // Two shader programs: emissive sun, lit planets
        GLuint sunProgram    = createProgramFromFiles("shaders/sun.vert",    "shaders/sun.frag");
        GLuint planetProgram = createProgramFromFiles("shaders/planet.vert", "shaders/planet.frag");

        SphereMesh sphere = createSphere();

        const glm::vec3 sunPos     = glm::vec3(0.0f);
        const glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
        const glm::vec3 earthColor = glm::vec3(0.0f, 0.0f, 0.0f);

        float lastTime = static_cast<float>(glfwGetTime());

        while (!glfwWindowShouldClose(window)) {
            float now = static_cast<float>(glfwGetTime());
            float dt  = now - lastTime;
            lastTime  = now;

            // WASD movement
            bool w = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
            bool a = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
            bool s = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
            bool d = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
            g_camera.processKeyboard(w, a, s, d, dt);

            glm::mat4 projection = glm::perspective(
                glm::radians(g_fov),
                (float)WINDOW_W / (float)WINDOW_H,
                NEAR_Z, FAR_Z
            );

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glm::mat4 view = g_camera.getViewMatrix();

            glBindVertexArray(sphere.vao);

            // -- Sun (emissive, no lighting) --
            glUseProgram(sunProgram);
            glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f));
            setSunUniforms(sunProgram, model, view, projection);
            glDrawElements(GL_TRIANGLES, sphere.indexCount, GL_UNSIGNED_INT, nullptr);

            // -- Earth (lit by the sun) --
            glUseProgram(planetProgram);
            float earthX = cos(now * 0.5f) * 20.0f;
            float earthZ = sin(now * 0.5f) * 20.0f;
            model = glm::translate(glm::mat4(1.0f), glm::vec3(earthX, 0.0f, earthZ));
            model = glm::rotate(model, now * 2.0f, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::scale(model, glm::vec3(2.0f));
            setPlanetUniforms(planetProgram, model, view, projection, earthColor, sunPos, lightColor);
            glDrawElements(GL_TRIANGLES, sphere.indexCount, GL_UNSIGNED_INT, nullptr);

            checkGLError("frame");

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        destroySphere(sphere);
        glDeleteProgram(sunProgram);
        glDeleteProgram(planetProgram);
        glfwTerminate();
    }
    catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        glfwTerminate();
        return -1;
    }

    return 0;
}
