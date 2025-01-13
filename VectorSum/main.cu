#include <iostream>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <random> 

using namespace std;

__global__ void Kernel(float* input, float* output, int n) {
    // Выделение общей памяти
    extern __shared__ float cache[];
    int i = threadIdx.x;
    int tid = i + blockIdx.x * blockDim.x;
    
    cache[i] = (tid < n) ? input[tid] : 0.0f;

    __syncthreads();

    // В цикле уменьшаем размер блока начиная с половины
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (i < s) {
            cache[i] += cache[i + s];
        }

        __syncthreads();
    }
    if (i == 0) {
        output[blockIdx.x] = cache[0];
    }
}

float VectorSumGPU(float* vec, int N) {
    float* d_input = nullptr;
    float* d_output = nullptr;

    int blockSize = 256;
    
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, numBlocks * sizeof(float));
    cudaMemcpy(d_input, vec, N * sizeof(float), cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();
    
    Kernel << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_input, d_output, N);
    
    cudaDeviceSynchronize();

    if (numBlocks > 1) {
        int remainingBlocks = (numBlocks + blockSize - 1) / blockSize;
        Kernel << <remainingBlocks, blockSize, blockSize * sizeof(float) >> > (d_output, d_output, numBlocks);
        cudaDeviceSynchronize();
        numBlocks = remainingBlocks;
    }

    float h_output = 0.0f;
    
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    auto diff = end - start;
    
    cout << "Time of work GPU function =" << chrono::duration<double, milli>(diff).count() << " ms" << endl;
    cout << "GPU result = " << h_output << endl;
    cudaFree(d_input);
    cudaFree(d_output);

    return h_output;
}

float VectorSumCPU(float* vec, int size) {
    float sum = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i){
        sum+=vec[i];
    }
    auto end = chrono::high_resolution_clock::now();
    auto diff = end - start;
    
    cout << "Time of work CPU function =" << chrono::duration<double, milli>(diff).count() << " ms" << endl;
    cout << "CPU result = " << sum << endl;
    return sum;
}

int main() {
    int size;
    cout << "Input size of matrix: ";
    cin >> size;
    float* vec = new float[size];
    for (int i = 0; i < size; ++i) {
        vec[i] = rand() % 10;
    }
    float cpu_sum = VectorSumCPU(vec, size);
    float gpu_sum = VectorSumGPU(vec, size);
    float diff = abs(cpu_sum - gpu_sum);
    cout << "Difference between CPU and GPU results = " << diff << endl;
    return 0;
}