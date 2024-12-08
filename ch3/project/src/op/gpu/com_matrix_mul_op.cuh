//
// Created by moguw on 24-12-8.
//

#ifndef COM_MATRIX_MUL_OP_CUH
#define COM_MATRIX_MUL_OP_CUH

namespace cudaop {
    void com_matrix_mul_op_cu(const float *A, const float *B, float *C, int row, int rol);
} // cudaop

#endif //COM_MATRIX_MUL_OP_CUH
