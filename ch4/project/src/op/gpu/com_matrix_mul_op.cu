//
// Created by moguw on 24-12-8.
//

#include "com_matrix_mul_op.cuh"

namespace cudaop {
    // 矩阵乘法CIDA内核函数：C = A * B
    __global__ void mmul(const float *A, const float *B, float *C, int ds) {
        int idx= blockIdx.x * blockDim.x + threadIdx.x;    // 计算当前线程的x索引（全局x坐标）
        int idy = blockIdx.y * blockDim.y + threadIdx.y;   // // 计算当前线程的y索引（全局y坐标）

        if ((idx < ds) && (idy < ds)) {
            float temp = 0;
            for (int i = 0; i < ds; i++)    // 对应行和列的点积操作
                temp += A[idx * ds + i] * B[i * ds + idy];    // 计算A的第idy行和B的第idx列的点积
            C[idx * ds + idy] = temp;
        }
    }

    void com_matrix_mul_op_cu(const float *A, const float *B, float *C, int row, int col) {
      const int block_size = 32;    // CUDA maximum is 1024 *total* threads in block

      dim3 block(block_size, block_size);    //  每个块包含32x32的线程

      dim3 grid((row+block.x-1) / block.x, (row+block.y-1)/block.y);

      mmul<<<grid, block>>>(A, B, C, row);    // 启动CUDA内核，执行矩阵乘法s
    }
} // cudaop