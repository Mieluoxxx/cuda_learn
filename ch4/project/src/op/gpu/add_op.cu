#include "add_op.cuh"

namespace cudaop {
    __global__ void add_op_cu_fp32(int32_t size, const int *in1, const int *in2, int *out) {
        int32_t global_index = threadIdx.x + blockDim.x * blockIdx.x;

        if (global_index >= size) {
            return;
        }

        float in_val1 = in1[global_index];
        float in_val2 = in2[global_index];

        out[global_index] = in_val1 + in_val2;
    }

    void add_op_cu(int *in1, int *in2, int *out, const int size) {
        int32_t thread_num = 512;
        int32_t block_num = (size + thread_num - 1) / thread_num;

        add_op_cu_fp32<<<block_num, thread_num>>>(size, in1, in2, out);
    }
}
