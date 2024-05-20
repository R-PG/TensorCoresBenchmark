#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>
#include <limits>
#include <mma.h>
#include <iostream>
#include <iomanip>
#include <sstream>

#ifndef MATRIXMUL_CUH
#define MATRIXMUL_CUH

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
struct MatrixMul
{
    MatrixType a[N][K];
    MatrixType b[K][M];

    MatrixMul();

    bool runTest();
};

std::ostream& operator<<(std::ostream& os, const half& value) {
    os << static_cast<float>(value); // Assuming half is convertible to float
    return os;
}

template<typename T, std::size_t D0, std::size_t D1>
void printMatrix(T (&a)[D0][D1])
{
    for(std::size_t i = 0; i < D0; ++i)
    {
        for(std::size_t j = 0; j < D1; ++j)
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

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
MatrixMul<MatrixType,AccumulatorType,M,N,K>::MatrixMul(){
    for (int i{0}; i < N; i++)
        for (int j{0}; j < K; j++)
            {   
                a[i][j] = (MatrixType)((double)((std::rand() % 4) + 1)/(double)((std::rand() % 6) + 1));
            }
    for (int i{0}; i < K; i++)
        for (int j{0}; j < M; j++)
            {
                b[i][j] = (MatrixType)((double)((std::rand() % 4) + 1)/(double)((std::rand() % 6) + 1));;
            }
    // {
    //     std::cout << "Matrix A generated: " << std::endl;
    //     printMatrix(a);
    //     std::cout << "Matrix B generated: " << std::endl;
    //     printMatrix(b);
    // }
};

template<typename T, std::size_t M, std::size_t N, std::size_t K>
__global__ void matrixMultiplicationKernel(T* A, T* B, T* C) // A way to adapt the type of teh regular operation to the type of the matrix operation?????? 
{
    int row = /*blockIdx.y * blockDim.y +*/ threadIdx.x;  // Compute row index
    int col = /*blockIdx.x * blockDim.x +*/ threadIdx.y;  // Compute column index

    // Check if within bounds
    //if (row < N && col < K) {
        T sum = 0;
        // Perform dot product of row of A and column of B to compute C[row][col]
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * M + col] = sum;
    //}
}

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
__global__ void wmma_ker(const MatrixType *a, const MatrixType *b, AccumulatorType *c)
{
    //auto w = warpSize;
    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, MatrixType, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, MatrixType, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, AccumulatorType> c_frag;
    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    // Load the inputs
    nvcuda::wmma::load_matrix_sync(a_frag, a, K);
    nvcuda::wmma::load_matrix_sync(b_frag, b, N);
    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // Store the output
    nvcuda::wmma::store_matrix_sync(c, c_frag, M, nvcuda::wmma::mem_row_major);
}

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
bool MatrixMul<MatrixType,AccumulatorType,M,N,K>::runTest() {
    MatrixType *d_a, *d_b;
    AccumulatorType *d_c;
    
    std::cout << "Starting test" << std::endl;

    // Allocate device memory
    checkCudaStatus(cudaMalloc((void **)&d_a, sizeof(MatrixType) * M * K));
    checkCudaStatus(cudaMalloc((void **)&d_b, sizeof(MatrixType) * K * N));
    checkCudaStatus(cudaMalloc((void **)&d_c, sizeof(AccumulatorType) * M * N));

    // Transfer data from host to device memory
    checkCudaStatus(cudaMemcpy(d_a, a, sizeof(MatrixType) * M * K, cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_b, b, sizeof(MatrixType) * K * N, cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemset(d_c, 0, M * N));

    //  Run nonTensorTest
    dim3 numBlocks(1);
    dim3 threadsPerBlock(M,N);
    ////////////////
    MatrixType *sameTypeDeviceC; // How to handle the difference of types?????
    checkCudaStatus(cudaMalloc((void **)&sameTypeDeviceC, sizeof(MatrixType) * N * M));
    checkCudaStatus(cudaMemset(sameTypeDeviceC, 0, M * N));
    ////////////////
    matrixMultiplicationKernel<MatrixType,M,N,K><<<numBlocks,threadsPerBlock>>>(d_a, d_b, sameTypeDeviceC);

    MatrixType nonTensorResult[N][M];
    checkCudaStatus(cudaMemcpy(nonTensorResult, sameTypeDeviceC, sizeof(MatrixType) * N * M, cudaMemcpyDeviceToHost));

    // std::cout << "Non Tensor Result: " << std::endl;
    // printMatrix(nonTensorResult);

    //  RunTensorTest
    checkCudaStatus(cudaMemset(d_c, 0, N * M));

    wmma_ker<MatrixType,AccumulatorType,M,N,K><<<numBlocks,threadsPerBlock>>>(d_a, d_b, d_c);

    AccumulatorType tensorResult[N][M];
    checkCudaStatus(cudaMemcpy(tensorResult, d_c, sizeof(AccumulatorType) * N * M, cudaMemcpyDeviceToHost));

    // std::cout << "Tensor result: " << std::endl;
    // printMatrix(tensorResult);

    bool passed = true;
    // Check Result
    for(int i = 0; i < M && passed; i++)
        for (int j = 0; j < N && passed; j++)
        {
            auto diff = (AccumulatorType) nonTensorResult[i][j] - tensorResult[i][j];
            if( diff > (AccumulatorType) 0.1)
            { 
                passed = false;
                std::cout << diff << std::endl;
                std::cout << nonTensorResult[i][j] << std::endl;
                std::cout << tensorResult[i][j] << std::endl;
            }
        }
    // Deallocate device memory
    checkCudaStatus(cudaFree(d_a));
    checkCudaStatus(cudaFree(d_b));
    checkCudaStatus(cudaFree(d_c));

    return passed;
}

#endif // MATRIXMUL_CUH
