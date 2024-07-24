#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "kernelHelpers.cuh"

//=========================================
/*               INTERFACE               */
//=========================================

/*
    All The matrices are considered to be in COLUMN MAJOR STORAGE
*/

inline void checkCudaStatus(cudaError_t status);

inline void checkCublasStatus(cublasStatus_t status);

template<typename T>
T* allocateDevice(std::size_t size, T *data = nullptr);

template<typename T>
void retrieveDevice(std::size_t size, T *data, T *devicePtr);

// Kernel definition for naive matrix multiplication
template <typename T>
__global__ void naiveMatrixMultiplyKernel(const T *a, const T *b, T *c, std::size_t m, std::size_t n ,std::size_t k);

// Host function to set up and call the kernel
template <typename T>
void naiveMatrixMultiply(const T *a, const T *b, T *c, std::size_t m, std::size_t n ,std::size_t k) ;

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
template<typename ABType, typename AccumulatorType>
__global__ void wmmaKernel(const ABType *a, const ABType *b, AccumulatorType *c, std::size_t m, std::size_t n ,std::size_t k);

template<typename ABType, typename AccumulatorType>
void wmmaMatrixMultiply(const ABType *a, const ABType *b, AccumulatorType *c, AccumulatorType alpha, AccumulatorType beta, std::size_t m, std::size_t n ,std::size_t k);

/// Wrapper executing floating point precision gemm with cublasLtMatmul
/// Workspace to support split-K algorithms
///
/// Matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed
template<cublasComputeType_t computeType, typename ABType, typename AccumulatorType>
void LtSgemm(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             size_t m,
             size_t n,
             size_t k,
             const AccumulatorType *alpha, /* host pointer */
             ABType *a,
             int lda,
             ABType *b,
             int ldb,
             const AccumulatorType *beta, /* host pointer */
             AccumulatorType *c,
             int ldc,
             void *workspace,
             size_t workspaceSize);

template<cublasComputeType_t ComputeType, typename ABType, typename AccumulatorType>
void cublasMatrixMultiply(const ABType* a, const ABType* b, AccumulatorType *c, std::size_t m, std::size_t n ,std::size_t k);

//=============================================
/*                IMPLEMENTATION              */
//=============================================

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
T* allocateDevice(std::size_t size, T *data)
{
    T *devicePtr;
    checkCudaStatus(cudaMalloc((void **)&devicePtr, sizeof(T) * size));
    if (data != nullptr)
        checkCudaStatus(cudaMemcpy(devicePtr, data, sizeof(T) * size, cudaMemcpyHostToDevice));
    else
        checkCudaStatus(cudaMemset(devicePtr, 0 , sizeof(T) * size));
    return devicePtr;
}

template<typename T>
void retrieveDevice(std::size_t size, T *data, T *devicePtr)
{
    checkCudaStatus(cudaMemcpy(data, devicePtr, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <typename T>
__global__ void naiveMatrixMultiplyKernel(const T *a, const T *b, T *c, std::size_t m, std::size_t n ,std::size_t k)
{
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        T acc{0};
        // Each thread computes one element of the block sub-matrix
        for (int i = 0; i < n; ++i) {
            //acc = __hfma_relu(a[row + i * m],b[i + col * n],acc);
            acc = a[row + i * m] * b[i + col * n] + acc;
        }
        c[row + col * m] = acc;
    }
}

template <typename T>
void naiveMatrixMultiply(const T *a, const T *b, T *c, std::size_t m, std::size_t n ,std::size_t k) 
{
    // Define the number of threads per block (16x16)
    dim3 threadsPerBlock(16, 16);
    // Calculate the number of blocks needed in the grid
    dim3 blocksPerGrid((k + 15) / 16, (m + 15) / 16);

    // Launch the kernel
    naiveMatrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, m, n, k);

    // Check for any errors launching the kernel
    checkCudaStatus(cudaGetLastError());
    
    // Wait for the GPU to finish
    checkCudaStatus(cudaDeviceSynchronize());
}

template<typename ABType, typename AccumulatorType>
__global__ void wmmaKernel(const ABType *a, const ABType *b, AccumulatorType *c, AccumulatorType alpha, AccumulatorType beta, std::size_t m, std::size_t n,std::size_t k)
{
    // WMMA dimensions
    static const size_t WMMA_M = kernelconfig::wmmaTileSize<ABType>::M;
    static const size_t WMMA_N = kernelconfig::wmmaTileSize<ABType>::N;
    static const size_t WMMA_K = kernelconfig::wmmaTileSize<ABType>::K;

    // Leading dimensions. Packed with no transpositions.
    int lda = m;
    int ldb = k;
    int ldc = m;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, ABType, nvcuda::wmma::col_major> aFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, ABType, nvcuda::wmma::col_major> bFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccumulatorType> accFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccumulatorType> cFrag;

    nvcuda::wmma::fill_fragment(accFrag, (AccumulatorType)0);

    // Loop over k
    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < m && aCol < k && bRow < k && bCol < n)
        {
            // Load the inputs
            nvcuda::wmma::load_matrix_sync(aFrag, a + aRow + aCol * lda, lda);
            nvcuda::wmma::load_matrix_sync(bFrag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    // Load in the current value of c and add this our result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < m && cCol < n)
    {
        //nvcuda::wmma::load_matrix_sync(cFrag, c + cRow + cCol * ldc, ldc, nvcuda::wmma::mem_col_major);

    // #pragma unroll
        //for(int i{0}; i < cFrag.num_elements; ++i)
            //cFrag.x[i] = alpha * accFrag.x[i] + beta * cFrag.x[i];

        // Store the output
        //nvcuda::wmma::store_matrix_sync(c + cRow + cCol * ldc, accFrag, ldc, nvcuda::wmma::mem_col_major);
        nvcuda::wmma::store_matrix_sync(c + cRow + cCol * ldc, accFrag, ldc, nvcuda::wmma::mem_col_major);
    }
}

template<typename ABType, typename AccumulatorType>
void wmmaMatrixMultiply(const ABType* a, const ABType* b, AccumulatorType *c, std::size_t m, std::size_t n ,std::size_t k) 
{
    // WMMA dimensions
    static const size_t WMMA_M = kernelconfig::wmmaTileSize<ABType>::M;
    static const size_t WMMA_N = kernelconfig::wmmaTileSize<ABType>::N;

    dim3 gridDim;
    dim3 blockDim;

    /*
        blockDim.x must be a multple of warpSize
        128x4 means we have 16 warps and a block computes a 64x64 output tile
    */
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (m + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (n + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    wmmaKernel<ABType,AccumulatorType><<<gridDim,blockDim>>>(a, b, c, 1, 0, m, n, k);

    // Check for any errors launching the kernel
    checkCudaStatus(cudaGetLastError());
    
    // Wait for the GPU to finish
    checkCudaStatus(cudaDeviceSynchronize());
}

template<cublasComputeType_t ComputeType, typename ABType, typename AccumulatorType>
void LtSgemm(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const AccumulatorType *alpha,
             const ABType *a,
             int lda,
             const ABType *b,
             int ldb,
             const AccumulatorType *beta,
             AccumulatorType *c,
             int ldc,
             void *workspace,
             size_t workspaceSize)
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, ComputeType, kernelconfig::cudaDataType<AccumulatorType>::t));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, kernelconfig::cudaDataType<ABType>::t, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, kernelconfig::cudaDataType<ABType>::t, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, kernelconfig::cudaDataType<AccumulatorType>::t, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha,
                                     a,
                                     Adesc,
                                     b,
                                     Bdesc,
                                     beta,
                                     c,
                                     Cdesc,
                                     c,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

template<cublasComputeType_t ComputeType, typename ABType, typename AccumulatorType>
void cublasMatrixMultiply(const ABType *a, const ABType *b, AccumulatorType *c, std::size_t m, std::size_t n ,std::size_t k)
{
    AccumulatorType alpha {1}, beta {0};
    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));
    
    void *workspace;
    size_t workspaceSize = m * n * 4;
    checkCudaStatus(cudaMalloc((void **)&workspace, workspaceSize));

    LtSgemm<ComputeType,ABType,AccumulatorType>(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a, m, b, k, &beta, c, m, workspace, workspaceSize);
    
    // Check for any errors launching the kernel
    checkCudaStatus(cudaGetLastError());
    
    // Wait for the GPU to finish
    checkCudaStatus(cudaDeviceSynchronize());

    checkCudaStatus(cudaFree(workspace));
}
#endif // KERNELS_CUH