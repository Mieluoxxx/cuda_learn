//
// Created by moguw on 24-12-10.
//

#ifndef VEC_ADD_OP_CUH
#define VEC_ADD_OP_CUH

namespace cudaop {
    template <typename T>
    void vec_add_op_cu(T* in1, T* in2, T* out, const int size);
}


#endif //VEC_ADD_OP_CUH
