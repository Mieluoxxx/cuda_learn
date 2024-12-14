//
// Created by moguw on 24-12-7.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>
#include "../utils.h"
#include "../../project/src/op/interface.h"

TEST(test_op, log_out) {
    printf("123456\n");
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
TEST(test_op, vec_add) {
    const int N = 32 * 1048576;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c; // CUDA 指针
    int size = N * sizeof(float);


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

    std::cout << std::endl;

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
