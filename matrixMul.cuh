#include <cublasLt.h>
#include "cublas_v2.h"
#include <cstdlib>
#include "helpers.h"
#include <limits>
#include <random>

#ifndef MATRIXMUL_CUH
#define MATRIXMUL_CUH

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N = M, std::size_t K = M>
struct MatrixMul
{
    using ResultType = typename hostResultType<MatrixType,AccumulatorType>::type;

    MatrixType **a = allocate2DArray<MatrixType>(M,K);
    MatrixType **b = allocate2DArray<MatrixType>(K,N);
    ResultType **naiveResult = allocate2DArray<ResultType>(M,N);
    ResultType **wmmaResult = allocate2DArray<ResultType>(M,N);
    ResultType **cublasResult = allocate2DArray<ResultType>(M,N);

    // WMMA dimensions
    static const size_t WMMA_M = wmmaTileSize<MatrixType,AccumulatorType>::M;
    static const size_t WMMA_N = wmmaTileSize<MatrixType,AccumulatorType>::N;
    static const size_t WMMA_K = wmmaTileSize<MatrixType,AccumulatorType>::K;

    MatrixMul();
    ~MatrixMul();

    void runTest(bool display = false);
};

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
MatrixMul<MatrixType,AccumulatorType,M,N,K>::MatrixMul()
{    
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    //std::uniform_real_distribution<> dis(std::numeric_limits<MatrixType>::min(), std::numeric_limits<MatrixType>::max());
    std::uniform_real_distribution<> dis(0, 9);

    for (int i{0}; i < K; i++)
        for (int j{0}; j < M; j++)
            {
                a[i][j] = (MatrixType) dis(gen);
            }
    for (int i{0}; i < N; i++)
        for (int j{0}; j < K; j++)
            {
                b[i][j] = (MatrixType) dis(gen);
            }
};

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
MatrixMul<MatrixType,AccumulatorType,M,N,K>::~MatrixMul()
{
    deallocate2DArray<MatrixType>(a,K);
    deallocate2DArray<MatrixType>(b,N);
    deallocate2DArray<ResultType>(naiveResult,N);
    deallocate2DArray<ResultType>(wmmaResult,N);
    deallocate2DArray<ResultType>(cublasResult,N);
}

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
__global__ void matrixMultiplicationKernel(MatrixType* A, MatrixType* B, AccumulatorType* C) // A way to adapt the type of teh regular operation to the type of the matrix operation?????? 
{
    int row = /*blockIdx.y * blockDim.y +*/ threadIdx.x;  // Compute row index
    int col = /*blockIdx.x * blockDim.x +*/ threadIdx.y;  // Compute column index

    // Check if within bounds
    //if (row < N && col < K) {
        MatrixType sum = 0;
        // Perform dot product of row of A and column of B to compute C[row][col]
        for (int i = 0; i < K; ++i) {
            sum += A[row + M * i] * B[col * K + i];
        }
        C[col * M + row] = (AccumulatorType) sum;
    //}
    // COLUMN MAYOR LAYOUT
}

// Kernel definition for matrix multiplication
template <typename T>
__global__ void matrixMultiplyKernel(const T* A, const T* B, T* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        T value = 0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

// Host function to set up and call the kernel
template <typename T>
void matrixMultiply(const T* A, const T* B, T* C, int n) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + 15) / 16, (n + 15) / 16);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n);
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K, 
std::size_t WMMA_M, std::size_t WMMA_N, std::size_t WMMA_K>
__global__ void wmma_ker(const MatrixType *a, const MatrixType *b, AccumulatorType *c, AccumulatorType alpha, 
AccumulatorType beta) 
{
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MatrixType, nvcuda::wmma::col_major> a_frag;
   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MatrixType, nvcuda::wmma::col_major> b_frag;
   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccumulatorType> acc_frag;
   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccumulatorType> c_frag;

   nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         nvcuda::wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      nvcuda::wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, nvcuda::wmma::mem_col_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      nvcuda::wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, nvcuda::wmma::mem_col_major);
   }
}

/// Sample wrapper executing single precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasSgemm,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
template<typename ScalarType, typename MatrixType, typename AccumulatorType>
void LtSgemm(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const ScalarType *alpha, /* host pointer */
             MatrixType *A,
             int lda,
             MatrixType *B,
             int ldb,
             const ScalarType *beta, /* host pointer */
             AccumulatorType *C,
             int ldc,
             void *workspace,
             size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, computeType<cudaType<MatrixType>::type,cudaType<AccumulatorType>::type>::type, 
                                                                cudaType<ScalarType>::type));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, cudaType<MatrixType>::type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, cudaType<MatrixType>::type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, cudaType<AccumulatorType>::type, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    //checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     beta,
                                     C,
                                     Cdesc,
                                     C,
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

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
void MatrixMul<MatrixType,AccumulatorType,M,N,K>::runTest(bool display) {
    MatrixType *d_a = NULL, *d_b = NULL;
    AccumulatorType *d_c = NULL, alpha {1}, beta {0};

    if (display)
    {
        std::cout << "Matrix A: " << std::endl;
        printMatrix<MatrixType>(a, M, K);
        std::cout << "Matrix B: " << std::endl;
        printMatrix<MatrixType>(b, K, N);
    }

    d_a = allocateDevice<MatrixType>(a, M, K);
    d_b = allocateDevice<MatrixType>(a, K, N);
    d_c = allocateDevice<AccumulatorType>(M, N);

    // Naive
    {
        matrixMultiply(d_a,d_b,d_c,M);
        retrieveDevice<ResultType>(naiveResult, d_c, M, N);
        
    }

    // WMMA
    {
        checkCudaStatus(cudaMemset(d_c, 0, M * N * sizeof(AccumulatorType)));

        dim3 gridDim;
        dim3 blockDim;

        /*
         blockDim.x must be a multple of warpSize
         128x4 means we have 16 warps and a block computes a 64x64 output tile
        */
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        wmma_ker<MatrixType,AccumulatorType,M,N,K,WMMA_M,WMMA_N,WMMA_K>
        <<<gridDim,blockDim>>>(d_a, d_b, d_c, alpha, beta);

        retrieveDevice<ResultType>(wmmaResult, d_c, M, N);

        if (display)
        {
            std::cout << "WMMA Result: " << std::endl;
            printMatrix<ResultType>(wmmaResult, M, N);
        }
    }

    // Cublas 
    {
        cublasLtHandle_t ltHandle;
        checkCublasStatus(cublasLtCreate(&ltHandle));
        
        void *workspace;
        size_t workspaceSize = M * N * 4;
        checkCudaStatus(cudaMalloc((void **)&workspace, workspaceSize));
        
        checkCudaStatus(cudaMemset(d_c, 0, M * N * sizeof(AccumulatorType)));

        LtSgemm<AccumulatorType,MatrixType,AccumulatorType>(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K, &beta, d_c, M, workspace, workspaceSize);
        
        checkCudaStatus(cudaFree(workspace));
        
        retrieveDevice<ResultType>(cublasResult, d_c, M, N);

        if (display)
        {   
            std::cout << "CuBLAS Result: " << std::endl;
            printMatrix<ResultType>(cublasResult, M, N);
        } 
    }

    bool passed = true;


    for(int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (naiveResult[i][j] != wmmaResult[i][j] || wmmaResult[i][j] != cublasResult[i][j]) 
            {
                passed = false;
            }
            if (display) std::cout << (cublasResult[i][j] != wmmaResult[i][j]);
        }
        if (display) std::cout << std::endl;
    }

    std::cout << "PASSED: " << passed << std::endl; 

    // Deallocate device memory
    checkCudaStatus(cudaFree(d_a));
    checkCudaStatus(cudaFree(d_b));
    checkCudaStatus(cudaFree(d_c));
}

#endif // MATRIXMUL_CUH
