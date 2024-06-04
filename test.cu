#include "matrixMul.cuh"
#define SEED 31

int main(){
    const size_t M = 64;
    const size_t N = 64;
    const size_t K = 64;
    
    {
        MatrixMul<half,float,M,N,K> test1;
        std::cout << "Half: " << M << "x" << N << "x" << K << " :Float" << std::endl;
        test1.runTest();
    }

    {
        MatrixMul<half,half,M,N,K> test2;
        std::cout << "Half: " << M << "x" << N << "x" << K << " :Half" << std::endl;
        test2.runTest(true);
    }

    {
        MatrixMul<double,double,M,N,K> test3;
        std::cout << "Double: " << M << "x" << N << "x" << K << " :Double" << std::endl;
        test3.runTest();
    }
}