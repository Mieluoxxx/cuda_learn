#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "base/base.h"
#include "gpu/add_operator.cuh"
#include "gpu/com_matrix_mul_op.cuh"

namespace mop{
    // 定义函数指针类型，用于指向加法内核函数
    typedef void (*AddKernel)(int* in1, int* in2, int* out, const int size);

    // 根据设备类型返回不同的加法内核
    AddKernel get_add_op(mbase::DeviceType device_type);
}

#endif
