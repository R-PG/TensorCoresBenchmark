cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Set the project name and enable CUDA
project(MatrixMultiplication LANGUAGES CXX CUDA)

# Specify the minimum required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the source files
set(SOURCES test.cu)

# Add the executable
add_executable(test ${SOURCES})

# Add CUDA compilation flags
set_target_properties(test PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# You can add more specific flags like this:
target_compile_options(test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-g -gencode arch=compute_86,code=sm_86>
)

# If you need to include additional directories
# target_include_directories(MatrixMul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# If you need to link additional libraries, use target_link_libraries
# target_link_libraries(MatrixMul PRIVATE some_library)