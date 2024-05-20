#include "matrixMul.cuh"
#define SEED 31

#define COMPUTE_CAPABILITY8 false

int main(){
    std::srand(SEED);

    MatrixMul<half,float,16,16,16> mul1;
    std::cout << "HalfxHalf=Float 16x16x16" << std::endl;
    if (mul1.runTest())
        std::cout << "PASSED" << std::endl; 
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl << std::endl;
    
    MatrixMul<half,float,32,8,16> mul2;
    std::cout << "HalfxHalf=Float 32x8x16" << std::endl;
    if (mul2.runTest())
        std::cout << "PASSED" << std::endl; 
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl << std::endl;

    MatrixMul<half,float,8,32,16> mul3;
    std::cout << "HalfxHalf=Float 8x32x16" << std::endl;
    if (mul3.runTest())
        std::cout << "PASSED" << std::endl; 
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl << std::endl;

    MatrixMul<half,half,16,16,16> mul4;
    std::cout << "HalfxHalf=Half 16x16x16" << std::endl;
    if (mul4.runTest())
        std::cout << "PASSED" << std::endl; 
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl << std::endl;

    MatrixMul<half,half,32,8,16> mul5;
    std::cout << "HalfxHalf=Half 32x8x16" << std::endl;
    if (mul5.runTest())
        std::cout << "PASSED" << std::endl; 
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl << std::endl;

    MatrixMul<half,half,8,32,16> mul6;
    std::cout << "HalfxHalf=Half 8x32x16" << std::endl;
    if (mul6.runTest())
        std::cout << "PASSED" << std::endl; 
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl << std::endl;

    MatrixMul<u_char,int,16,16,16> mul7;
    std::cout << "UnsignedxUnsigned=Int 16x16x16" << std::endl;
    if (mul3.runTest())
        std::cout << "PASSED" << std::endl; 
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl << std::endl;

    if constexpr (1)
    {
        MatrixMul<double,double,8,8,4> mul1;
        std::cout << "Double 8x8x4" << std::endl;
        if (mul1.runTest())
            std::cout << "PASSED" << std::endl; 
        else
            std::cout << "FAILED" << std::endl;
        std::cout << std::endl << std::endl;
    }
    return 0;
}