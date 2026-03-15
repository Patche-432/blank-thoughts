#include "shader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        throw std::runtime_error(std::string("Shader compile error: ") + log);
    }

    return shader;
}

GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER,   vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        throw std::runtime_error(std::string("Program link error: ") + log);
    }

    glDeleteShader(vert);
    glDeleteShader(frag);

    return program;
}

GLuint createProgramFromFiles(const std::string& vertPath, const std::string& fragPath) {
    auto readFile = [](const std::string& path) -> std::string {
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("Could not open shader file: " + path);
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    };

    std::string vertSrc = readFile(vertPath);
    std::string fragSrc = readFile(fragPath);

    return createProgram(vertSrc.c_str(), fragSrc.c_str());
}
