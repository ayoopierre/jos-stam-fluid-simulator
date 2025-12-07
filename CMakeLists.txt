cmake_minimum_required(VERSION 3.18) # 3.18+ recommended for good CUDA support
project(MyProject LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include directory for headers
include_directories(${PROJECT_SOURCE_DIR}/inc)

# Gather source files (.cpp and .cu automatically)
file(GLOB_RECURSE SOURCES
     "${PROJECT_SOURCE_DIR}/src/*.cpp"
     "${PROJECT_SOURCE_DIR}/src/*.cu"
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Enable separable compilation for CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: Print out the sources for debugging
message(STATUS "Sources: ${SOURCES}")
