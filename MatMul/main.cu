#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

using namespace std;

// Функция ядра
__global__ void Kernel(const float* A, const float* B, float* C, int size) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < size && column < size) {
        int result = 0;
        for (int i = 0; i < size; ++i) {
            result += A[row * size + i] * B[i * size + column];
        }
        C[row * size + column] = result;
    }
}

// Функция умножения матриц на GPU
void MatMulGPU(const float* A, const float* B, float* C, int size) {
    float* A_copy, * B_copy, * C_copy;
    size_t matrix_size = size * size * sizeof(float);
    // Выделение памяти
    cudaMalloc((void**)&A_copy, matrix_size);
    cudaMalloc((void**)&B_copy, matrix_size);
    cudaMalloc((void**)&C_copy, matrix_size);
    // Копирование входных данных с CPU на GPU
    cudaMemcpy(A_copy, A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_copy, B, matrix_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // Вызов функции ядра
    Kernel << <blocksPerGrid, threadsPerBlock >> > (A_copy, B_copy, C_copy, size);
    // Копирование результата с GPU на CPU
    cudaMemcpy(C, C_copy, matrix_size, cudaMemcpyDeviceToHost);
    // Освобождение памяти на GPU
    cudaFree(A_copy);
    cudaFree(B_copy);
    cudaFree(C_copy);
}

// Функция умножения матриц на СPU
void MatMulCPU(const float* A, const float* B, float* C, int size) {
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
            C[row * size + column] = 0;
            for (int i = 0; i < size; ++i) {
                C[row * size + column] += A[row * size + i] * B[i * size + column];
            }
        }
    }
}

// Функция сравнения матриц с погрешностью
bool MatCompare(const float* A, const float* B, int size, float eps) {
    for (int i = 0; i < size * size; ++i) {
        if (abs(A[i] - B[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    int size;
    cout << "Input size of matrix: ";
    cin >> size;
    float* A = new float[size * size];
    float* B = new float[size * size];
    float* C_GPU = new float[size * size];
    float* C_CPU = new float[size * size];
    float eps = 0.0001;
    for (int i = 0; i < size * size; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    auto start = chrono::high_resolution_clock::now();
    MatMulCPU(A, B, C_CPU, size);
    auto end = chrono::high_resolution_clock::now();
    auto diff = end - start;
    
    cout << "Time of work CPU function =" << chrono::duration<double, milli>(diff).count() << " ms" << endl;

    start = chrono::high_resolution_clock::now();
    MatMulGPU(A, B, C_GPU, size);
    end = chrono::high_resolution_clock::now();
    diff = end - start;
    cout << "Time of work CGU function =" << chrono::duration<double, milli>(diff).count() << " ms" << endl;

    cout << C_GPU[0] << " " << C_CPU[0] << endl;
    cout << C_GPU[size] << " " << C_CPU[size] << endl;
    cout << C_GPU[size*size-1] << " " << C_CPU[size*size-1] << endl;
    
    if (MatCompare(C_GPU, C_CPU, size, eps)) {
        cout << "Results are correct!" << endl;
    }
    else {
        cout << "Results are incorrect!" << endl;
    }
    delete[] A;
    delete[] B;
    delete[] C_GPU;
    delete[] C_CPU;
    return 0;
}