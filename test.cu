#include "helpers.cuh"

int main()
{

    println("Initiating matrix multiplication test");
    println("Operation C = A * B with A, B and C square matrices of dimension N and '*' matrix multipication\n");
    
    println("Half to Half:");
    {
        using ABType = half;
        using ResultType = half;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_16F;
        for (auto i{256}; i <= 16384; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            if (naive == wmma && wmma == cublas)
                println("\t\t\033[31mResults differ\033[0m");
            else
                println("\t\t\033[32mIdentic results on the three methods\033[0m");
        }
    }

    println("Half to Float:");
    {
        using ABType = half;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
        for (auto i{256}; i <= 16384; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            if (wmma == cublas)
                println("\t\t\033[31mCuBlas and WMMA results differ\033[0m");
            else
                println("\t\t\033[32mCuBlas and WMMA results are identic\033[0m");

            if (elemWiseEqual(naive, cublas))
                println("\t\t\033[31mNavie and cuBLAS results differ\033[0m");
            else
                println("\t\t\033[32mNaive and cuBLAS results are identic\033[0m");
        }
    }

    println("BFloat16 to Float:");
    {
        using ABType = nv_bfloat16;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
        for (auto i{256}; i <= 16384; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            if (wmma == cublas)
                println("\t\t\033[31mCuBlas and WMMA results differ\033[0m");
            else
                println("\t\t\033[32mCuBlas and WMMA results are identic\033[0m");

            if (elemWiseEqual(naive, cublas))
                println("\t\t\033[31mNavie and cuBLAS results differ\033[0m");
            else
                println("\t\t\033[32mNaive and cuBLAS results are identic\033[0m");
        }
    }

    println("Float to Float no intermediate reduced precision:");
    {
        using ABType = float;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
        for (auto i{256}; i <= 16384; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            println("\t\tWMMA API...");
            //auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            // if (wmma == cublas)
            //     println("\t\t\033[31mCuBlas and WMMA results differ\033[0m");
            // else
            //     println("\t\t\033[32mCuBlas and WMMA results are identic\033[0m");

            if (elemWiseEqual(naive, cublas))
                println("\t\t\033[31mNavie and cuBLAS results differ\033[0m");
            else
                println("\t\t\033[32mNaive and cuBLAS results are identic\033[0m");
        }
    }
     
    println("Double to Double:");
    {
        using ABType = double;
        using ResultType = double;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_64F;
        for (auto i{256}; i <= 16384; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            if (wmma == cublas)
                println("\t\t\033[31mCuBlas and WMMA results differ\033[0m");
            else
                println("\t\t\033[32mCuBlas and WMMA results are identic\033[0m");

            if (elemWiseEqual(naive, cublas))
                println("\t\t\033[32mNaive and cuBLAS results are identic\033[0m");
            else
                println("\t\t\033[31mNavie and cuBLAS results differ\033[0m");
        }
    }
}