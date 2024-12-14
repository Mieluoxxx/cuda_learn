//
// Created by moguw on 24-12-14.
//

#include "shared_matmul.cuh"

namespace cudaop {
    template<typename T>
    __global__ void matmul_com(const T *A, const T *B, T *C, int ds) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        if ((idx < ds) && (idy < ds)) {
            T sum = 0;
            for (int i = 0; i < ds; i++) {
                sum += A[idy * ds + i] * B[i * ds + idx];
            }
            C[idy * ds + idx] = sum;
        }
    }

    template<typename T>
    __global__ void matmul(const T *A, const T *B, T *C, int ds, int block_size) {
        extern __shared__ char shared_mem[];

        T *As = (T *) shared_mem;
        T *Bs = (T *) &As[block_size * block_size];

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if ((idx < ds) && (idy < ds)) {
            T temp = 0;
            for (int i = 0; i < ds / block_size; i++) {
                As[threadIdx.y * block_size + threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
                Bs[threadIdx.y * block_size + threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

                __syncthreads();

                for (int k = 0; k < block_size; k++) {
                    temp += As[threadIdx.y * block_size + k] * Bs[k * block_size + threadIdx.x];
                }

                __syncthreads();
            }

            C[idy * ds + idx] = temp;
        }
    }

    template void cudaop::com_matmul_op_cu<float>(const float *A, const float *B, float *C, int row, int col);

    template void cudaop::com_matmul_op_cu<double>(const double *A, const double *B, double *C, int row, int col);

    template void cudaop::com_matmul_op_cu<int>(const int *A, const int *B, int *C, int row, int col);

    template<typename T>
    void com_matmul_op_cu(const T *A, const T *B, T *C, int row, int col) {
        int block_size = 3;
        size_t shared_mem_size = 2 * block_size * block_size * sizeof(T);

        dim3 block(block_size, block_size);
        dim3 grid((row + block.x - 1) / block.x, (row + block.y - 1) / block.y);

        matmul<T><<<grid, block, shared_mem_size>>>(A, B, C, row, block_size);
    }

    // -------------- 矩阵行和 --------------
    template<typename T>
    __global__ void mat_row_sum(const T *A, T *sum, size_t ds) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < ds) {
            T temp = 0;
            for (size_t i = 0; i < ds; i++) {
                temp += A[idx * ds + i];
            }
            sum[idx] = temp;
        }
    }

    template void cudaop::mat_row_sum_op_cu<float>(const float *A, float *sum, const int row);

    template void cudaop::mat_row_sum_op_cu<double>(const double *A, double *sum, const int row);

    template void cudaop::mat_row_sum_op_cu<int>(const int *A, int *sum, const int row);

    template<typename T>
    void mat_row_sum_op_cu(const T *A, T *sum, const int row) {
        int block_size = 256;
        int grid = (row + block_size - 1) / block_size;
        mat_row_sum<T><<<grid, block_size>>>(A, sum, row);
    }

    // -------------- 矩阵列和 --------------
    template<typename T>
    __global__ void mat_col_sum(const T *A, T *sum, size_t ds) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < ds) {
            T temp = 0;
            for (size_t i = 0; i < ds; i++) {
                temp += A[i * ds + idx];
            }
            sum[idx] = temp;
        }
    }

    template void cudaop::mat_col_sum_op_cu<float>(const float *A, float *sum, const int col);

    template void cudaop::mat_col_sum_op_cu<double>(const double *A, double *sum, const int col);

    template void cudaop::mat_col_sum_op_cu<int>(const int *A, int *sum, const int col);

    template<typename T>
    void mat_col_sum_op_cu(const T *A, T *sum, const int col) {
        int block_size = 256;
        int grid = (col + block_size - 1) / block_size;
        mat_col_sum<T><<<grid, block_size>>>(A, sum, col);
    }
}
