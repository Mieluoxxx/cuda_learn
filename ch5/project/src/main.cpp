#define THREADS_PRE_BLOCK 512

#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "op/interface.h"

void generateRandomArray(int *&a, int n) {
    a = new int[n];
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < n; ++i) {
        a[i] = rand() % 100;
    }
}

void checkCudaError(cudaError_t err, const char *action) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during: " << action << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
  *
  * 程序的入口点。
  * 分配 GPU 和 CPU 内存。
  * 生成两个随机整数数组 a 和 b。
  * 将这些数组从 CPU 内存复制到 GPU 内存。
  * 调用 CUDA 内核函数（通过 mop::get_add_op 获取）来执行加法运算。
  * 检查内核执行是否有错误。
  * 将结果从 GPU 内存复制回 CPU 内存。
  * 打印出结果数组 c 的前 10 个元素。
  * 释放分配的内存。
 */
int main(int argc, char *argv[]) {
    const int N = 1024;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c; // CUDA 指针
    int size = N * sizeof(float); // 向量维度

    // 分配 CPU 内存
    a = (float *) malloc(size);
    b = (float *) malloc(size);
    c = (float *) malloc(size);

    checkCudaError(cudaMalloc((void **) (&d_a), size), "cudaMalloc for d_a");
    checkCudaError(cudaMalloc((void **) (&d_b), size), "cudaMalloc for d_b");
    checkCudaError(cudaMalloc((void **) (&d_c), size), "cudaMalloc for d_c");

    // 生成随机数组
    // generateRandomArray(a, N);
    // generateRandomArray(b, N);

    for (int i = 0; i < N; i++) {
        // initialize vectors in host memory
        a[i] = rand() / (float) RAND_MAX;
        b[i] = rand() / (float) RAND_MAX;
        c[i] = 0;
    }

    // 将数据从 CPU 复制到 GPU
    checkCudaError(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice), "cudaMemcpy for d_a");
    checkCudaError(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice), "cudaMemcpy for d_b");

    // 调用 CUDA 内核，传递 N 而不是 size
    mop::get_vec_add_op<float>(mbase::DeviceType::Device)(d_a, d_b, d_c, N);

    // 检查内核执行错误
    checkCudaError(cudaGetLastError(), "Kernel execution");

    // 将结果从 GPU 复制到 CPU
    checkCudaError(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy for d_c");

    // 打印部分结果
    for (int i = 0; i < N; ++i) {
        if (c[i] != (a[i] + b[i])) {
            std::cout << "Vec A :" << a[i] << std::endl;
            std::cout << "Vec B :" << b[i] << std::endl;
            std::cout << "Vec C :" << c[i] << std::endl;
        }
    }


    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
