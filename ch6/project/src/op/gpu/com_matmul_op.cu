//
// Created by moguw on 24-12-8.
//

#include "com_matmul_op.cuh"

namespace cudaop {
    // 矩阵乘法CIDA内核函数：C = A * B
    __global__ void matmul_com(const float *A, const float *B, float *C, int ds) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程的x索引（全局x坐标）
        int idy = blockIdx.y * blockDim.y + threadIdx.y; // 计算当前线程的y索引（全局y坐标）
        if ((idx < ds) && (idy < ds)) {
            float temp = 0;
            for (int i = 0; i < ds; i++) // 对应行和列的点积操作
                temp += A[idx * ds + i] * B[i * ds + idy]; // 计算A的第idy行和B的第idx列的点积
            C[idx * ds + idy] = temp;
        }
    }

    __global__ void matmul(const float *A, const float *B, float *C, int ds, int block_size) {
        extern __shared__ float shared_mem[];

        float *As = shared_mem;
        float *Bs = &shared_mem[block_size * block_size];

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if ((idx < ds) && (idy < ds)) {
            float temp = 0;
            for (int i = 0; i < ds / block_size; i++) {
                As[threadIdx.y * block_size + threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
                Bs[threadIdx.y * block_size + threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

                __syncthreads();

                for (int k = 0; k < block_size; k++)
                    temp += As[threadIdx.y * block_size + k] * Bs[k * block_size + threadIdx.x];

                __syncthreads();
            }

            C[idy * ds + idx] = temp;
        }
    }

    void com_matmul_op_cu(const float *A, const float *B, float *C, int row, int col) {
        int block_size = 3;
        size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);

        dim3 block(block_size, block_size);
        dim3 grid((row + block.x - 1) / block.x, (col + block.y - 1) / block.y);

        matmul<<<grid, block, shared_mem_size>>>(A, B, C, row, block_size);
    }
} // cudaop
