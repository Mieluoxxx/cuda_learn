//
// Created by moguw on 24-12-10.
//

#include "vec_add_op.cuh"

namespace cudaop {
    template<typename T>
    __global__ void vec_add_op_native(const T *in1, const T *in2, T *out, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            out[idx] = in1[idx] + in2[idx];
        }
    }

    // grid-stride loop
    // vector add kernel: C = A+B
    template<typename T>
    __global__ void vec_add_op(const T *A, const T *B, T *C, int ds) {
        for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < ds; idx += blockDim.x * gridDim.x) {
            C[idx] = A[idx] + B[idx];
        }
    }

    // 显式实例化模板
    template void cudaop::vec_add_op_cu<float>(float *in1, float *in2, float *out, const int size);

    template void cudaop::vec_add_op_cu<double>(double *in1, double *in2, double *out, const int size);

    template void cudaop::vec_add_op_cu<int>(int *in1, int *in2, int *out, const int size);

    template<typename T>
    void vec_add_op_cu(T *in1, T *in2, T *out, const int size) {
        int32_t block_num = 32;
        int32_t thread_num = 32;

        vec_add_op<<<block_num, thread_num>>>(in1, in2, out, size);
    }
}
