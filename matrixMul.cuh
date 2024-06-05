#include <cublasLt.h>
#include "cublas_v2.h"
#include <cstdlib>
#include "helpers.h"
#include <limits>
#include <random>

#ifndef MATRIXMUL_CUH
#define MATRIXMUL_CUH

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
struct MatrixMul
{
    using ResultType = typename hostResultType<MatrixType,AccumulatorType>::type;

    MatrixType **a = allocate2DArray<MatrixType>(M,K);
    MatrixType **b = allocate2DArray<MatrixType>(K,N);
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

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

/// Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers
///int8_t
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const MatrixType *A,
                   int lda,
                   const MatrixType *B,
                   int ldb,
                   AccumulatorType *C,
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
        size_t workspaceSize = 1024 * 1024 * 4;
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
            if (cublasResult[i][j] != wmmaResult[i][j]) 
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
