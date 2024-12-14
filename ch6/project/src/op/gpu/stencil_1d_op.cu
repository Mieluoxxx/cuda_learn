//
// Created by moguw on 24-12-9.
//

#include "stencil_1d_op.cuh"
#include <stdio.h>

namespace cudaop {
    __global__ void stencil_1d_op(const int *in, int *out, int block_size, int padding) {
        extern __shared__ int temp[]; // 动态分配的共享内存
        int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
        int lidx = threadIdx.x + padding;

        // 读取数据到共享内存
        temp[lidx] = in[g_idx];

        if (threadIdx.x < padding) {
            temp[lidx - padding] = in[g_idx - padding]; // 读取左边界
            temp[lidx + block_size] = in[g_idx + block_size]; // 读取右边界
        }

        // 同步
        __syncthreads();

        // 应用stencil操作
        int result = 0;
        for (int offset = -padding; offset <= padding; offset++) {
            result += temp[lidx + offset];
        }

        // 写回结果
        out[g_idx] = result;
    }


    // 一维模板stencil
    void stencil_1d_op_cu(int *in, int *out, int arraySize, int padding) {
        int block_size = 8;
        int grid_size = arraySize / block_size;

        printf("arraySize: %d, grid_size: %d, padding: %d\n", arraySize, grid_size, padding);

        int shared_mem_size = block_size + 2 * padding;

        stencil_1d_op<<<grid_size, block_size, shared_mem_size>>>(in + padding, out + padding, block_size, padding);
    }
}

