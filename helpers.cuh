#ifndef HELPERS_CUH
#define HELPERS_CUH

#include <cstring>
#include <iostream>
#include <iomanip>
#include <functional>
#include "kernels.cuh"
#include <memory>
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

    Matrix_2D();
    Matrix_2D(size_t rows, size_t cols);
    Matrix_2D(Matrix_2D &&);
    Matrix_2D&& operator=(Matrix_2D&&);
    ~Matrix_2D();
    
    Matrix_2D(const Matrix_2D&) = delete;
    Matrix_2D& operator=(const Matrix_2D&) = delete;

    template<typename R>
    bool operator==(const Matrix_2D<R>&) const;

    void deviceSync();

    T& elem(size_t row, size_t col);
 
    const T& elem(size_t row, size_t col) const;

    void print() const;

    void fillRandom();
};

template <typename T>
T double2Type(double);

template <typename T>
double type2Double(T);

template<typename ABType, typename AccumulatorType>
Matrix_2D<AccumulatorType> matrixMulNaive(const Matrix_2D<ABType> a, const Matrix_2D<ABType> b);

template<typename ABType, typename ResultType>
Matrix_2D<ResultType> matrixMulWMMA(const Matrix_2D<ABType>& a, const Matrix_2D<ABType>& b);

template<cublasComputeType_t ComputeType, typename ABType, typename ResultType>
Matrix_2D<ResultType> matrixMulCuBLAS(const Matrix_2D<ABType>& a, const Matrix_2D<ABType>& b);

template<typename T, typename R>
bool elemWiseMatrixCompare(const Matrix_2D<T>& a, const Matrix_2D<T>& b);

void println(const char* str);

void println(std::string str);

void print(const char* str);
void print(std::string str);

//=============================================
/*                IMPLEMENTATION              */
//=============================================

template <typename T>
Matrix_2D<T>::Matrix_2D() {}

template <typename T>
Matrix_2D<T>::Matrix_2D(Matrix_2D&& other) : 
    _rows(other._rows), _cols(other._cols), _data(std::move(other._data)), _device(other._device) {}

template <typename T>
Matrix_2D<T>&& Matrix_2D<T>::operator=(Matrix_2D<T>&& other)
{
    _cols = std::move(other._cols);
    _rows = std::move(other._rows);
    _data = std::move(other._data);
    _device = std::move(other._device);
}

template <typename T>
Matrix_2D<T>::Matrix_2D(size_t rows, size_t cols) 
    : _rows(rows), _cols(cols), _data(std::make_unique<T[]>(rows * cols)), _device(allocateDevice<T>(rows * cols)) {}

template <typename T>
Matrix_2D<T>::~Matrix_2D()
{
    checkCudaStatus(cudaFree(_device));
} 

template <typename T>
template<typename R>
bool Matrix_2D<T>::operator==(const Matrix_2D<R>& other) const 
{
    return (_cols != other._cols || _rows != other._rows) ? 
        false :
        !std::memcmp(_data.get(),other._data.get(),_rows * _cols * std::min(sizeof(T),sizeof(R)));
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
const T& Matrix_2D<T>::elem(size_t col, size_t row) const
{
    return _data[col * _cols + row];
}

template <typename T>
T double2Type(double elem) {return elem;} 

template <>
half double2Type(double item) {return __double2half(item);}

template <>
__nv_bfloat16 double2Type(double item) {return __double2bfloat16(item);}

template <typename T>
void Matrix_2D<T>::fillRandom()
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1);

    for (int i{0}; i < _cols; ++i)
        for (int j{0}; j < _rows; ++j)
            elem(i,j) = double2Type<T>(dis(gen));

    checkCudaStatus(cudaMemcpy(_device, _data.get(), sizeof(T) * _rows * _cols, cudaMemcpyHostToDevice));
}

template <typename T>
double type2Double(T item) {return item;} 

template <>
double type2Double(half item) {return __half2float(item);} 

template <typename T>
void Matrix_2D<T>::print() const
{
    for (int j = 0; j < _rows; ++j) {
        for (int i = 0; i < _cols; ++i) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(10) << type2Double(elem(i,j)) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename ABType>
Matrix_2D<ABType> matrixMulNaive(const Matrix_2D<ABType>& a, const Matrix_2D<ABType>& b)
{
    Matrix_2D<ABType> result(a._rows, b._cols);
    naiveMatrixMultiply(a._device, b._device, result._device, a._rows, b._cols, a._cols);
    result.deviceSync();
    return result;
}

template<typename ABType, typename ResultType>
Matrix_2D<ResultType> matrixMulWMMA(const Matrix_2D<ABType>& a, const Matrix_2D<ABType>& b)
{
    Matrix_2D<ResultType> result(a._rows, b._cols);
    wmmaMatrixMultiply(a._device, b._device, result._device, a._rows, b._cols, a._cols);
    result.deviceSync();
    return result;
}

template<cublasComputeType_t ComputeType, typename ABType, typename ResultType>
Matrix_2D<ResultType> matrixMulCuBLAS(const Matrix_2D<ABType>& a, const Matrix_2D<ABType>& b)
{
    Matrix_2D<ResultType> result(a._rows, b._cols);
    cublasMatrixMultiply<ComputeType,ABType,ResultType>(a._device, b._device, result._device, a._rows, b._cols, a._cols);
    result.deviceSync();
    return result;
}

template<typename T, typename R>
bool elemWiseEqual(const Matrix_2D<T>& a, const Matrix_2D<R>& b)
{
    if (a._cols != b._cols || a._rows != b._rows) return false;
    for (int i{0}; i < a._cols; ++i)
        for (int j{0}; j < a._rows; ++j)
            if (a.elem(i,j) - ((T) b.elem(i,j)) > (T) 0) return false;
    return true;
}

template <typename T, typename R>
double subs(T a, R b) {return type2Double(a) - type2Double(b);} 

template<typename T, typename R>
double maxEps(const Matrix_2D<R>& a, const Matrix_2D<T>& b)
{
    double maxEps = 0;
    if (a._cols != b._cols || a._rows != b._rows) return -1;
    for (int i{0}; i < a._cols; ++i)
        for (int j{0}; j < a._rows; ++j)
        {
            double eps = std::abs(subs(a.elem(i,j),b.elem(i,j)));
            maxEps = std::fmax(maxEps,eps);
        }
    return maxEps;
}

void println(const char* str) {std::cout << str << std::endl;}

void println(std::string str) {std::cout << str << std::endl;}

void print(const char* str) {std::cout << str;}

void print(std::string str) {std::cout << str;}

#endif // HELPERS_CUH