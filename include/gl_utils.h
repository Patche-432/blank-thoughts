#pragma once

#include <string>

// Checks and prints any pending OpenGL errors.
// Call after draw calls or state changes for debugging.
void checkGLError(const std::string& where);
