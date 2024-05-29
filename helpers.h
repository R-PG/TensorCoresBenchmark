#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <sstream>

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