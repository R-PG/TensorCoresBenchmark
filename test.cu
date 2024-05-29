#include "matrixMul.cuh"
#define SEED 31

#define COMPUTE_CAPABILITY8 false

int main(){
    // MatrixMul<half,float,16,16,16> test1;
    // std::cout << "HalfxHalf=Float 16x16x16" << std::endl;
    // test1.runTest(true);

    // MatrixMul<half,float,32,8,16> test2;
    // std::cout << "HalfxHalf=Float 32x8x16" << std::endl;
    // test2.runTest();

    // MatrixMul<half,float,8,32,16> test3;
    // std::cout << "HalfxHalf=Float 8x32x16" << std::endl;
    // test3.runTest();

    // MatrixMul<half,half,16,16,16> test4;
    // std::cout << "HalfxHalf=Half 16x16x16" << std::endl;
    // test4.runTest();

    // MatrixMul<half,half,32,8,16> test5;
    // std::cout << "HalfxHalf=Half 32x8x16" << std::endl;
    // test5.runTest();

    // MatrixMul<half,half,8,32,16> test6;
    // std::cout << "HalfxHalf=Half 8x32x16" << std::endl;
    // test6.runTest();

    // MatrixMul<int8_t,int,32,8,16> test7;
    // std::cout << "UnsignedxUnsigned=Int 16x16x16" << std::endl;
    // test7.runTest();

    if constexpr (0)
    {
        MatrixMul<double,double,8,8,4> test8;
        std::cout << "Double 8x8x4" << std::endl;
        test8.runTest(true);
    }

    MatrixMul<float,float,4,4,4> test1;
    std::cout << "HalfxHalf=Float 16x16x16" << std::endl;
    test1.runTest(true);
    return 0;
}