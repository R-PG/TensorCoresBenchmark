#include "matrixMul.cuh"

int main(){
    using MatrixType = half;
    using AccumulatorType = half;

    if constexpr(0)
    {
        {
            const size_t N = 256;
            MatrixMul<MatrixType,AccumulatorType,N> test;
            test.runTest();
        }
        
        {
            const size_t N = 512;
            MatrixMul<MatrixType,AccumulatorType,N> test;
            test.runTest();
        }

        {
            const size_t N = 1024;
            MatrixMul<MatrixType,AccumulatorType,N> test;
            test.runTest();
        }

        {
            const size_t N = 2048;
            MatrixMul<MatrixType,AccumulatorType,N> test;
            test.runTest();
        }   

        {
            const size_t N = 4096;
            MatrixMul<MatrixType,AccumulatorType,N> test;
            test.runTest();
        }
    }

    if constexpr(1)
    {
        {
            const size_t N = 8192;
            MatrixMul<MatrixType,AccumulatorType,N> test;
            test.runTest();
        }

        {
            const size_t N = 16384;
            MatrixMul<MatrixType,AccumulatorType,N> test;
            test.runTest();
        }
    }
}