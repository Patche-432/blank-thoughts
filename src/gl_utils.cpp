#include "gl_utils.h"

#include <glad/glad.h>
#include <iostream>

void checkGLError(const std::string& where) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "[OpenGL Error] " << where << " : " << err << std::endl;
    }
}
