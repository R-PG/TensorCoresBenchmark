#include "helpers.cuh"

int main()
{
    size_t maxN = 16384;
    println("Initiating matrix multiplication test");
    println("Operation C = A * B with A, B and C square matrices of dimension N and '*' matrix multipication\n");
    
    println("Half to Half:");
    {
        using ABType = half;
        using ResultType = half;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_16F;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();
            
            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);

            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Half naive and Half cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Half naive and Half cublas: " << eps << "\033[0m" << std::endl;

            eps = maxEps(wmma, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
        }
    }

    println("Half to Float:");
    {
        using ABType = half;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();

            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            
            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Float naive and Half cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Float naive and Half cublas: " << eps << "\033[0m" << std::endl;

            eps = maxEps(wmma, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
        }
    }

    println("BFloat16 to Float:");
    {
        using ABType = nv_bfloat16;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();

            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            
            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Float naive and BFloat16 cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Float naive and BFloat16 cublas: " << eps << "\033[0m" << std::endl;

            eps = maxEps(wmma, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
        }
    }

    println("Double to Double:");
    {
        using ABType = double;
        using ResultType = double;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_64F;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();

            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);
            
            println("\t\tWMMA API...");
            auto wmma = matrixMulWMMA<ABType, ResultType>(a, b);
            
            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);
            
            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Double naive and Double cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Double naive and Double cublas: " << eps << "\033[0m" << std::endl;

            eps = maxEps(wmma, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between wmma and cublas: " << eps << "\033[0m" << std::endl;
        }
    }

    println("Float to Float; no intermediate reduced precision accelerator:");
    {
        using ABType = float;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();

            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);

            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);

            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
        }
    }

    println("Float to Float; Half intermediate precision accelerator:");
    {
        using ABType = float;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F_FAST_16F;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();

            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);

            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);

            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
        }
    }

    println("Float to Float; BFloat16 intermediate precision accelerator:");
    {
        using ABType = float;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F_FAST_16BF;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();

            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);

            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);

            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
        }
    }

    println("Float to Float; TF32 intermediate precision accelerator:");
    {
        using ABType = float;
        using ResultType = float;
        const cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        for (auto i{256}; i <= maxN; i = i << 1)
        {
            println("\tN = " + std::to_string(i));
            Matrix_2D<ABType> a(i, i), b(i, i);
            a.fillRandom();
            b.fillRandom();
            
            println("\t\tNaive implementation...");
            auto naive = matrixMulNaive(a, b);

            println("\t\tCuBLAS...");
            auto cublas = matrixMulCuBLAS<ComputeType, ABType, ResultType>(a, b);

            auto eps = maxEps(naive, cublas);
            if (eps == 0)
                std::cout << "\t\t\033[33mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
            else
                std::cout << "\t\t\033[31mMaximum epsilon between Float naive and Float cublas: " << eps << "\033[0m" << std::endl;
        }
    }
}