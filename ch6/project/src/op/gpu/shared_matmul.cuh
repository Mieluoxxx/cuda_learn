//
// Created by moguw on 24-12-14.
//

#ifndef SHARED_MATMUL_CUH
#define SHARED_MATMUL_CUH

namespace cudaop {
    template <typename T>
    void com_matmul_op_cu(const T* A, const T* B, T *C, int row, int col);

    template <typename T>
    void mat_row_sum_op_cu(const T* A, T *sum, const int row);

    template <typename T>
    void mat_col_sum_op_cu(const T* A, T *sum, const int col);
}

#endif //SHARED_MATMUL_CUH
