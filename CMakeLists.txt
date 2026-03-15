cmake_minimum_required(VERSION 3.20)
project(SolarSystem C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------
# Find dependencies
# -------------------------
find_package(OpenGL REQUIRED)
find_package(glfw3  REQUIRED)
find_package(glm    REQUIRED)

# -------------------------
# GLAD as its own C library
# -------------------------
add_library(glad STATIC vendor/glad/src/glad.c)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)
target_include_directories(glad PUBLIC vendor/glad/include)

# -------------------------
# Executable
# -------------------------
add_executable(SolarSystem
    src/main.cpp
    src/gl_utils.cpp
    src/shader.cpp
    src/sphere_mesh.cpp
    src/camera.cpp
)

target_include_directories(SolarSystem PRIVATE
    include
    vendor/glad/include
)

target_link_libraries(SolarSystem PRIVATE
    glad
    glfw
    glm::glm
    OpenGL::GL
)

# -------------------------
# Copy shaders next to the binary after build
# -------------------------
add_custom_command(TARGET SolarSystem POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:SolarSystem>/shaders
    COMMENT "Copying shaders to output directory"
)
