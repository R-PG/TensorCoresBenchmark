#include "helpers.cuh"

int main(){

    Matrix_2D<half> a(16,16);
    a.fillRandom();
    Matrix_2D<half> b(16,16);
    b.fillRandom();
    auto naive  = matrixMulNaive(a,b);
    auto wmma  = matrixMulWMMA<half,half>(a,b);
    auto cublas  = matrixMulCuBLAS<CUBLAS_COMPUTE_16F,half,half>(a,b);
    a.print();
    std::cout << std::endl;
    b.print();
    std::cout << std::endl;
    naive.print();
    std::cout << std::endl;
    wmma.print();
    std::cout << std::endl;
    cublas.print();
    std::cout << std::endl;
}