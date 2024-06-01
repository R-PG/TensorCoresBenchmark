#include <cublasLt.h>
#include "cublas_v2.h"
#include <cstdlib>
#include "helpers.h"
#include <limits>
#include <mma.h>
#include <random>

#ifndef MATRIXMUL_CUH
#define MATRIXMUL_CUH

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
struct MatrixMul
{
    MatrixType a[M][K];
    MatrixType b[K][N];

    MatrixMul();

    void runTest(bool display = false);
};

template<typename MatrixType, typename AccumulatorType, std::size_t M, std::size_t N, std::size_t K>
MatrixMul<MatrixType,AccumulatorType,M,N,K>::MatrixMul(){    
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    //std::uniform_real_distribution<> dis(std::numeric_limits<MatrixType>::min(), std::numeric_limits<MatrixType>::max());
    std::uniform_real_distribution<> dis(0, 9);

    for (int i{0}; i < M; i++)
        for (int j{0}; j < K; j++)
            {
                a[i][j] = (MatrixType) dis(gen);
            }
    for (int i{0}; i < K; i++)
        for (int j{0}; j < N; j++)
            {
                b[i][j] = (MatrixType) dis(gen);
            }
};

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
    AccumulatorType *d_c = NULL;

    if (display)
    {
        std::cout << "Matrix A: " << std::endl;
        printMatrix(a);
        std::cout << "Matrix B: " << std::endl;
        printMatrix(b);
    }
    
    d_a = allocateCMLDevice(a);
    d_b = allocateCMLDevice(b);
    d_c = allocateCMLDevice<AccumulatorType,M,N>();

    dim3 numBlocks(1);
    dim3 threadsPerBlock(M,N);

    //  Run nonTensorTest
    //if constexpr(1){
        matrixMultiplicationKernel<MatrixType,AccumulatorType,M,N,K><<<numBlocks,threadsPerBlock>>>(d_a, d_b, d_c);

        AccumulatorType nonTensorResult[M][N];
        retrieveCMLDevice(nonTensorResult, d_c);
        
        if (display)
        {
            std::cout << "Regular Kernel Result: " << std::endl;
            printMatrix(nonTensorResult);
        }
    //}

    threadsPerBlock = dim3(1,32);
    //  WMMA Test
    //if constexpr(1){   
        checkCudaStatus(cudaMemset(d_c, 0, M * N));

        wmma_ker<MatrixType,AccumulatorType,M,N,K><<<numBlocks,threadsPerBlock>>>(d_a, d_b, d_c);

        AccumulatorType wmmaResult[M][N];
        retrieveCMLDevice(wmmaResult, d_c);

        if (display)
        {
            std::cout << "WMMA Result: " << std::endl;
            printMatrix(wmmaResult);
        }
    //}

    // Cublas
    if constexpr(1)
    {   
        cublasLtHandle_t ltHandle;
        checkCublasStatus(cublasLtCreate(&ltHandle));
        // AccumulatorType cublasResult[M * N];
        // //LtIgemmTensor<MatrixType,AccumulatorType,M,N,K>(ltHandle,M,N,K,a,M,b,K,cublasResult,M);
        
        const AccumulatorType alpha {1.f};
        const AccumulatorType beta {0.f};
        void *workspace;
        size_t workspaceSize = 1024 * 1024 * 4;
        checkCudaStatus(cudaMalloc((void **)&workspace, workspaceSize));
        checkCudaStatus(cudaMemset(d_c, 0, M * N));

        LtSgemm<AccumulatorType,MatrixType,AccumulatorType>(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K, &beta, d_c, M, workspace, workspaceSize);
        checkCudaStatus(cudaFree(workspace));
        
        AccumulatorType cublasResult[M][N];
        retrieveCMLDevice(cublasResult, d_c);
        if (display)
        {   
            std::cout << "WMMA Result: " << std::endl;
            printMatrix(cublasResult);
        }
    }    

    AccumulatorType residualMatrix[M][N];
    auto& result = wmmaResult;
    
    for(int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            residualMatrix[i][j] = (AccumulatorType) nonTensorResult[i][j] - result[i][j];

    for(int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (residualMatrix[i][j] != 0) 
                std::cout << i << "," << j << ": " << residualMatrix[i][j] << std::endl;

    // Deallocate device memory
    checkCudaStatus(cudaFree(d_a));
    checkCudaStatus(cudaFree(d_b));
    checkCudaStatus(cudaFree(d_c));
}

#endif // MATRIXMUL_CUH
