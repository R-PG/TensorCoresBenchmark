#include "matrixMul.cuh"
#define SEED 31

#define COMPUTE_CAPABILITY8 false

int main(){
    MatrixMul<half,float,16,16,16> test1;
    std::cout << "HalfxHalf=Float 16x16x16" << std::endl;
    test1.runTest();

    MatrixMul<half,float,32,8,16> test2;
    std::cout << "HalfxHalf=Float 32x8x16" << std::endl;
    test2.runTest();

    MatrixMul<half,float,8,32,16> test3;
    std::cout << "HalfxHalf=Float 8x32x16" << std::endl;
    test3.runTest();

    MatrixMul<half,half,16,16,16> test4;
    std::cout << "HalfxHalf=Half 16x16x16" << std::endl;
    test4.runTest();

    MatrixMul<half,half,32,8,16> test5;
    std::cout << "HalfxHalf=Half 32x8x16" << std::endl;
    test5.runTest();

    MatrixMul<half,half,8,32,16> test6;
    std::cout << "HalfxHalf=Half 8x32x16" << std::endl;
    test6.runTest();

    MatrixMul<double,double,8,8,4> test8;
    std::cout << "Double 8x8x4" << std::endl;
    test8.runTest();
}