#ifndef KERNELHELPERS_CUH
#define KERNELHELPERS_CUH

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace kernelconfig
{
/*
    Get tile sizes
*/
template<typename ABType>
struct wmmaTileSize {};

template<>
struct wmmaTileSize<half>
{
    static const size_t M = 16;
    static const size_t N = 16;
    static const size_t K = 16;
};

template<>
struct wmmaTileSize<nv_bfloat16> 
{
    static const size_t M = 16;
    static const size_t N = 16;
    static const size_t K = 16;
};

// template<>
// struct wmmaTileSize<nvcuda::wmma::precision::tf32> 
// {
//     static const size_t M = 16;
//     static const size_t N = 16;
//     static const size_t K = 8;
// };

template<>
struct wmmaTileSize<double> 
{
    static const size_t M = 8;
    static const size_t N = 8;
    static const size_t K = 4;
};

/*
    Get compile time cudaDataType_t enum associated with the type
*/

template<typename T>
struct cudaDataType {};

template<>
struct cudaDataType<half> {static const cudaDataType_t t = CUDA_R_16F;};

template<>
struct cudaDataType<nv_bfloat16> {static const cudaDataType_t t = CUDA_R_16BF;};

template<>
struct cudaDataType<float> {static const cudaDataType_t t = CUDA_R_32F;};

template<>
struct cudaDataType<double> {static const cudaDataType_t t = CUDA_R_64F;};
} //    END namespace kernelconfig
#endif //KERNELHELPERS_CUH