#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <mma.h>
#include <sstream>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))  // Operator to convert: Column Mayor Layout INDEXING -> Row Mayor Storage

std::ostream& operator<<(std::ostream& os, const half& value) {
    os << static_cast<float>(value); // Assuming half is convertible to float
    return os;
}

std::ostream& operator<<(std::ostream& os, const int8_t& value) {
    os << static_cast<int>(value); // Assuming half is convertible to float
    return os;
}

template<typename T, std::size_t M, std::size_t N>
void printMatrix(const T(&a)[M][N])
{
    for(std::size_t i = 0; i < M; ++i)
    {
        for(std::size_t j = 0; j < N; ++j)
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

template<typename T, std::size_t M, std::size_t N>
inline T* allocateCMLDevice(const T(&a)[M][N])
{ 
    T temp[M * N], *d;
    checkCudaStatus(cudaMalloc((void **)&d, sizeof(T) * M * N));
    for (int i{0}; i < M; ++i)
        for (int j{0}; j < N; ++j)
            temp[IDX2C(i,j,M)] = a[i][j];
    checkCudaStatus(cudaMemcpy(d, temp, sizeof(T) * M * N, cudaMemcpyHostToDevice));
    return d;
}

template<typename T, std::size_t M, std::size_t N>
T* allocateRMLDevice(const T(&a)[M][N])
{
    T temp[M * N], *d;
    checkCudaStatus(cudaMalloc((void **)&d, sizeof(T) * M * N));
    for (int i{0}; i < M; ++i)
        for (int j{0}; j < N; ++j)
            temp[i * M + j] = a[i][j];
    checkCudaStatus(cudaMemcpy(d, temp, sizeof(T) * M * N, cudaMemcpyHostToDevice));
    return d;
}

template<typename T, std::size_t M, std::size_t N>
T* allocateCMLDevice()
{
    T* d;
    checkCudaStatus(cudaMalloc((void **)&d, sizeof(T) * M * N));
    checkCudaStatus(cudaMemset(d, 0, M * N));
    return d;
}

template<typename T, std::size_t M, std::size_t N>
void retrieveCMLDevice(T(&a)[M][N], T* d)
{
    T temp[M * N];
    checkCudaStatus(cudaMemcpy(temp, d, sizeof(T) * M * N, cudaMemcpyDeviceToHost));
    for (int i{0}; i < M; ++i)
        for (int j{0}; j < N; ++j)
            a[i][j] = temp[j * M + i];
}

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