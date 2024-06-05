#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <mma.h>
#include <sstream>

template <typename T>
T** allocate2DArray(size_t rows, size_t cols) {
    // Allocate memory for an array of pointers (each pointing to a col)
    T** array = new T*[cols];

    // Allocate a single block of memory for all the elements
    T* data = new T[rows * cols];

    // Set the row pointers to point to the appropriate positions in the block
    for (int i = 0; i < cols; ++i) {
        array[i] = data + i * rows;
    }

    return array;
}

template <typename T>
void deallocate2DArray(T** array, size_t cols) {
    // Deallocate the block of memory containing all the elements
    delete[] array[0];

    // Deallocate the array of pointers
    delete[] array;
}

std::ostream& operator<<(std::ostream& os, const half& value) {
    os << static_cast<float>(value); // Assuming half is convertible to float
    return os;
}

std::ostream& operator<<(std::ostream& os, const int8_t& value) {
    os << static_cast<int>(value); // Assuming half is convertible to float
    return os;
}

template<typename T>
void printMatrix(T **a, std::size_t M, std::size_t N)
{
    for(std::size_t i = 0; i < N; ++i)
    {
        for(std::size_t j = 0; j < M; ++j)
        {
            std::cout << std::setw(10) << std::fixed << std::setprecision(10) << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cout << "cuda API failed with status " << status << ": " << cudaGetErrorString(status) << std::endl;
        throw std::logic_error("cuda API failed");
    }
};

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

template<typename T>
T* allocateDevice(T **a, std::size_t M, std::size_t N)
{
    T *d;
    checkCudaStatus(cudaMalloc((void **)&d, sizeof(T) * M * N));
    checkCudaStatus(cudaMemcpy(d, a[0], sizeof(T) * M * N, cudaMemcpyHostToDevice));
    return d;
}

template<typename T>
T* allocateDevice(std::size_t M, std::size_t N)
{
    T* d;
    checkCudaStatus(cudaMalloc((void **)&d, sizeof(T) * M * N));
    checkCudaStatus(cudaMemset(d, 0, M * N));
    return d;
}

// template<typename T = float>
// void retrieveDevice(float **a, half* d, std::size_t M, std::size_t N)
// {
//     half **temp = allocate2DArray<half>(M,N);
//     checkCudaStatus(cudaMemcpy(temp, d, sizeof(half) * M * N, cudaMemcpyDeviceToHost));
//     for (int i{0}; i < N; ++i)
//         for (int j{0}; j < M; ++j)
//             a[i][j] = __half2float(temp[i][j]);
//     deallocate2DArray<half>(temp,M);
// }

template<typename T>
void retrieveDevice(T **a, T* d, std::size_t M, std::size_t N)
{
    checkCudaStatus(cudaMemcpy(a[0], d, sizeof(T) * M * N, cudaMemcpyDeviceToHost));
}

/*
    Get comparator type
*/
template<typename MatrixType, typename AccumulatorType>
struct hostResultType {};

template<>
struct hostResultType<half,half> {using type = half;};

template<>
struct hostResultType<half,float> {using type = float;};

template<>
struct hostResultType<double,double> {using type = double;};

/*
    Get compile time compute type based on the Scale Type and Atype/Btype 
*/
template<cudaDataType_t ScaleType, cudaDataType_t ABtype>
struct computeType {};

template<>
struct computeType<CUDA_R_16F,CUDA_R_32F> {static const cublasComputeType_t type = CUBLAS_COMPUTE_32F;};

template<>
struct computeType<CUDA_R_16F,CUDA_R_16F> {static const cublasComputeType_t type = CUBLAS_COMPUTE_16F;};

template<>
struct computeType<CUDA_R_64F,CUDA_R_64F> {static const cublasComputeType_t type = CUBLAS_COMPUTE_64F;};

template<typename T>
struct cudaType {};

template<>
struct cudaType<half> {static const cudaDataType_t type = CUDA_R_16F;};

template<>
struct cudaType<float> {static const cudaDataType_t type = CUDA_R_32F;};

template<>
struct cudaType<double> {static const cudaDataType_t type = CUDA_R_64F;};

template<typename MatrixType, typename AccumulatorType>
struct wmmaTileSize {};

/*
    Get tile sizes
*/

template<>
struct wmmaTileSize<half,float> 
{
    static const size_t M = 16;
    static const size_t N = 16;
    static const size_t K = 16;
};

template<>
struct wmmaTileSize<half,half> 
{
    static const size_t M = 16;
    static const size_t N = 16;
    static const size_t K = 16;
};

template<>
struct wmmaTileSize<__nv_bfloat16,float> 
{
    static const size_t M = 16;
    static const size_t N = 16;
    static const size_t K = 16;
};

// template<>
// struct wmmaTileSize<nvcuda::wmma::precision::tf32,float> 
// {
//     static const size_t M = 16;
//     static const size_t N = 16;
//     static const size_t K = 8;
// };

template<>
struct wmmaTileSize<double,double> 
{
    static const size_t M = 8;
    static const size_t N = 8;
    static const size_t K = 4;
};