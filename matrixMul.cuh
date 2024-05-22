#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublasLt.h>
#include "cublas_v2.h"
#include <cstdlib>
#include <limits>
#include <mma.h>
#include <iostream>
#include <iomanip>
#include <sstream>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#ifndef MATRIXMUL_CUH
#define MATRIXMUL_CUH

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
struct MatrixMul
{
    MatrixType a[M * K];
    MatrixType b[K * N];

    MatrixMul();

    void runTest(bool display = false);
};

std::ostream& operator<<(std::ostream& os, const half& value) {
    os << static_cast<float>(value); // Assuming half is convertible to float
    return os;
}

std::ostream& operator<<(std::ostream& os, const int8_t& value) {
    os << static_cast<int>(value); // Assuming half is convertible to float
    return os;
}

template<typename T, std::size_t N>
void printMatrix(const T(&a)[N], std::size_t ld)
{
    for(std::size_t i = 0; i < ld; ++i)
    {
        for(std::size_t j = 0; j < N/ld; ++j)
        {
            std::cout << std::setw(10) << std::fixed << std::setprecision(10) << a[IDX2C(i,j,ld)] << " ";
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

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
MatrixMul<MatrixType,AccumulatorType,M,N,K>::MatrixMul(){
    for (int i{0}; i < M; i++)
        for (int j{0}; j < K; j++)
            {   
                a[IDX2C(i,j,M)] = (MatrixType)((double)((std::rand() % 4) + 1)/(double)((std::rand() % 6) + 1));
            }
    for (int i{0}; i < K; i++)
        for (int j{0}; j < N; j++)
            {
                b[IDX2C(i,j,K)] = (MatrixType)((double)((std::rand() % 4) + 1)/(double)((std::rand() % 6) + 1));;
            }
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
            sum += A[row + i * M] * B[col * K + i];
        }
        C[row + col * M] = sum;
    //}
    // COLUMN MAYOR LAYOUT
}

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
__global__ void wmma_ker(const MatrixType *a, const MatrixType *b, AccumulatorType *c)
{
    //auto w = warpSize;
    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, MatrixType, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, MatrixType, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, AccumulatorType> c_frag;
    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    // Load the inputs
    nvcuda::wmma::load_matrix_sync(a_frag, a, M);
    nvcuda::wmma::load_matrix_sync(b_frag, b, K);
    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // Store the output
    nvcuda::wmma::store_matrix_sync(c, c_frag, M, nvcuda::wmma::mem_col_major);
}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

/// Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers
///
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const int8_t *A,
                   int lda,
                   const int8_t *B,
                   int ldb,
                   int32_t *C,
                   int ldc) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t *Atransform = NULL, *Btransform = NULL;
    int32_t *Ctransform                   = NULL;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform));

    checkCublasStatus(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    // tensor op igemm kernels only support NT gemm
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for original matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // data memory order is set to CUBLASLT_ORDER_COL4_4R2_8C in order to achieve best performance on Turing devices.
    // for best performance on Ampere, consider setting the memory order to CUBLASLT_ORDER_COL32_2R_4R4.
    checkCublasStatus(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // ---------------------------------------------------------------------------------------------
    // transforms and computation

    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0));

    // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0));

    // no need to transform C matrix as beta is assumed to be 0
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     Atransform,
                                     AtransformDesc,
                                     Btransform,
                                     BtransformDesc,
                                     &beta,
                                     Ctransform,
                                     CtransformDesc,
                                     Ctransform,
                                     CtransformDesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    opTranspose = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    // transform outputs to COL order
    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CtransformDesc, &transformBeta, NULL, NULL, C, Cdesc, 0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (CtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(CtransformDesc));
    if (BtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(BtransformDesc));
    if (AtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(AtransformDesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if (transformDesc) checkCublasStatus(cublasLtMatrixTransformDescDestroy(transformDesc));

    // wait until device is done before freeing transformed buffers
    checkCudaStatus(cudaDeviceSynchronize());
    if (Ctransform) checkCudaStatus(cudaFree(Ctransform));
    if (Btransform) checkCudaStatus(cudaFree(Btransform));
    if (Atransform) checkCudaStatus(cudaFree(Atransform));
}


template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
void MatrixMul<MatrixType,AccumulatorType,M,N,K>::runTest(bool display) {
    MatrixType *d_a, *d_b;
    AccumulatorType *d_c;

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
    checkCudaStatus(cudaMalloc((void **)&sameTypeDeviceC, sizeof(MatrixType) * M * N));
    checkCudaStatus(cudaMemset(sameTypeDeviceC, 0, M * N));
    ////////////////
    matrixMultiplicationKernel<MatrixType,M,N,K><<<numBlocks,threadsPerBlock>>>(d_a, d_b, sameTypeDeviceC);

    MatrixType nonTensorResult[M * N];
    checkCudaStatus(cudaMemcpy(nonTensorResult, sameTypeDeviceC, sizeof(MatrixType) * M * N, cudaMemcpyDeviceToHost));

    //  WMMA Test
    checkCudaStatus(cudaMemset(d_c, 0, M * N));

    wmma_ker<MatrixType,AccumulatorType,M,N,K><<<numBlocks,threadsPerBlock>>>(d_a, d_b, d_c);

    AccumulatorType tensorResult[M * N];
    checkCudaStatus(cudaMemcpy(tensorResult, d_c, sizeof(AccumulatorType) * M * N, cudaMemcpyDeviceToHost));

    // Cublas
    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));
    AccumulatorType cublasResult[M * N];
    LtIgemmTensor(ltHandle,M,N,K,a,M,b,K,cublasResult,M);

    auto& result = cublasResult;
    AccumulatorType residualMatrix[M * N];
    // Check Result
    for(int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            residualMatrix[IDX2C(i,j,M)] = (AccumulatorType) nonTensorResult[IDX2C(i,j,M)] - result[IDX2C(i,j,M)];
        }

    if (display)
    {
        std::cout << "Matrix A: " << std::endl;
        printMatrix(a,M);
        std::cout << "Matrix B: " << std::endl;
        printMatrix(b,K);
        std::cout << "Regular Kernel Result: " << std::endl;
        printMatrix(nonTensorResult,M);
        std::cout << "wmma result: " << std::endl;
        printMatrix(tensorResult,M);
        std::cout << "CUBLAS result: " << std::endl;
        printMatrix(cublasResult,M);
        std::cout << "Residual matrix: " << std::endl;
        printMatrix(residualMatrix,M);
        std::cout << std::endl << std::endl;
    }
    // Deallocate device memory
    checkCudaStatus(cudaFree(d_a));
    checkCudaStatus(cudaFree(d_b));
    checkCudaStatus(cudaFree(d_c));
}

#endif // MATRIXMUL_CUH
