//
// Created by moguw on 24-12-8.
//

#ifndef COM_MATMUL_OP_CUH
#define COM_MATMUL_OP_CUH

namespace cudaop {
    void com_matmul_op_cu(const float *A, const float *B, float *C, int row, int rol);
} // cudaop

#endif //COM_MATMUL_OP_CUH
