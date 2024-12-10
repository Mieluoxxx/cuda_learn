//
// Created by moguw on 24-12-9.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "utils.h"
#include "../../project/src/op/interface.h"
#include <algorithm>
#include <cuda_runtime_api.h>

void printArray(int *array, int size) {
    for(int i = 0; i < size; i++){
        printf("%d ", array[i]);
    }
    printf("\n");
}

void fill_ints(int *x, int n) {
    std::fill_n(x, n, 1);
}

TEST(test_op, op_stencil_1d) {
    int arraySize = 8;

    int padding = 3;
    int real_size = arraySize + 2 * padding;
    int* h_in;
    int* h_out;

    // 分配 CPU 内存
    h_in = (int*)malloc(real_size * sizeof(int));
    h_out = (int*)malloc(real_size * sizeof(int));

    fill_ints(h_in, real_size);
    fill_ints(h_out, real_size);

    // 设备数组
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, real_size * sizeof(int));
    cudaMalloc((void**)&d_out, real_size * sizeof(int));

    // 将数据从 CPU 复制到 GPU
    cudaMemcpy(d_in, h_in, real_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, real_size * sizeof(int), cudaMemcpyHostToDevice);

    // 计算
    mop::get_stencil_1d_op(mbase::DeviceType::Device)(d_in, d_out, arraySize, padding);

    // 将结果从 GPU 复制到 CPU
    cudaMemcpy(h_out, d_out, real_size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印输出
    printf("Input Array:\n");
    printArray(h_in, real_size);

    printf("Output Array:\n");
    printArray(h_out, real_size);

    // 释放内存
    free(h_in);
    free(h_out);

    cudaFree(d_in);
    cudaFree(d_out);
}
