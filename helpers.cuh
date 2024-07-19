#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <iostream>
#include <iomanip>
#include "kernels.cuh"
#include <memory.h>
#include <random>
#include <sstream>

//=========================================
/*               INTERFACE               */
//=========================================

/*
    Matrix stored in dynamic memory and cuda device memory
    COLUMN MAJOR LAYOUT
*/
template <typename T>
struct Matrix_2D
{
    std::unique_ptr<T[]> _data;
    T* _device;
    size_t _rows, _cols;
    
    Matrix_2D() = delete;
    Matrix_2D(const Matrix_2D&) = delete;
    Matrix_2D& operator= (const Matrix_2D&) = delete;
    Matrix_2D&& operator= (const Matrix_2D&&) = delete;
    
    Matrix_2D(Matrix_2D &&);
    Matrix_2D(size_t rows, size_t cols);
    ~Matrix_2D();

    void deviceSync();

    T& elem(size_t row, size_t col);

    void print();

    void fillRandom();
};

template<typename ABType, typename AccumulatorType>
Matrix_2D<AccumulatorType> matrixMulNaive(Matrix_2D<ABType> a, Matrix_2D<ABType> b);

//=============================================
/*                IMPLEMENTATION              */
//=============================================

template <typename T>
Matrix_2D<T>::Matrix_2D(Matrix_2D&& other) : 
    _data(std::move(other._data)), _device(other._device), _rows(other._rows), _cols(other._cols) {}

template <typename T>
Matrix_2D<T>::Matrix_2D(size_t rows, size_t cols) 
    : _rows(rows), _cols(cols), _data(std::make_unique<T[]>(rows * cols)), _device(allocateDevice<T>(rows * cols)) {};

template <typename T>
Matrix_2D<T>::~Matrix_2D()
{
    checkCudaStatus(cudaFree(_device));
} 

template <typename T>
void Matrix_2D<T>::deviceSync()
{
    retrieveDevice(_rows * _cols, _data.get(), _device);
}

template <typename T>
T& Matrix_2D<T>::elem(size_t col, size_t row)
{
    return _data[col * _cols + row];
}

template <typename T>
void Matrix_2D<T>::print() 
{
    for (int j = 0; j < _rows; ++j) {
        for (int i = 0; i < _cols; ++i) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(10) << elem(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

template <>
void Matrix_2D<half>::print() 
{
    for (int j = 0; j < _rows; ++j) {
        for (int i = 0; i < _cols; ++i) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(10) << __half2float(elem(i,j)) << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void Matrix_2D<T>::fillRandom()
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 10);

    for (int i{0}; i < _cols; i++)
        for (int j{0}; j < _rows; j++)
            elem(i,j) = (T) dis(gen);

    checkCudaStatus(cudaMemcpy(_device, _data.get(), sizeof(T) * _rows * _cols, cudaMemcpyHostToDevice));
}

template<typename ABType>
Matrix_2D<ABType> matrixMulNaive(Matrix_2D<ABType>& a, Matrix_2D<ABType>& b)
{
    Matrix_2D<ABType> result(a._rows, b._cols);
    naiveMatrixMultiply(a._device, b._device, result._device, a._rows, b._cols, a._cols);
    result.deviceSync();
    return result;
};

template<typename ABType, typename AccumulatorType>
Matrix_2D<AccumulatorType> matrixMulWMMA(Matrix_2D<ABType>& a, Matrix_2D<ABType>& b)
{
    Matrix_2D<AccumulatorType> result(a._rows, b._cols);
    wmmaMatrixMultiply(a._device, b._device, result._device, a._rows, b._cols, a._cols);
    result.deviceSync();
    return result;
};

template<cublasComputeType_t ComputeType, typename ABType, typename AccumulatorType>
Matrix_2D<AccumulatorType> matrixMulCuBLAS(Matrix_2D<ABType>& a, Matrix_2D<ABType>& b)
{
    Matrix_2D<AccumulatorType> result(a._rows, b._cols);
    cublasMatrixMultiply<ComputeType,ABType,AccumulatorType>(a._device, b._device, result._device, a._rows, b._cols, a._cols);
    result.deviceSync();
    return result;
};

#endif // HELPERS_CUH