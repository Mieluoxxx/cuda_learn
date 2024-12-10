//
// Created by moguw on 24-12-9.
//

#ifndef STENCIL_1D_OP_CUH
#define STENCIL_1D_OP_CUH

namespace cudaop {
    void stencil_1d_op_cu(int *in, int *out, int arraySize, int padding);
}

#endif //STENCIL_1D_OP_CUH
