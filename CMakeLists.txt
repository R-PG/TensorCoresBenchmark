cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Set the project name and enable CUDA
project(test LANGUAGES CXX CUDA)

# Specify the minimum required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the source files
set(SOURCES kernels.cu helpers.cu test.cu)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add CUDA compilation flags
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# You can add more specific flags like this:
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-g -rdc=true -gencode arch=compute_86,code=sm_86>
)

set(CUDART_LIBRARY cudart)
set(CUBLASLT_LIBRARY cublasLt)

if(NOT WIN32)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

target_link_libraries(${PROJECT_NAME}
    ${CUDART_LIBRARY}
    ${CUBLASLT_LIBRARY}
)