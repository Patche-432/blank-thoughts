#pragma once

#include <glad/glad.h>
#include <string>

// Compiles a single shader stage from source string.
// Throws std::runtime_error on failure.
GLuint compileShader(GLenum type, const char* src);

// Links a compiled vertex + fragment shader into a program.
// Throws std::runtime_error on failure.
GLuint createProgram(const char* vertSrc, const char* fragSrc);

// Loads shader source from file paths and creates a program.
// Throws std::runtime_error if files cannot be read or shaders fail.
GLuint createProgramFromFiles(const std::string& vertPath, const std::string& fragPath);
