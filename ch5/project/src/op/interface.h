#ifndef INTERFACE_H_
#define INTERFACE_H_

#include "base/base.h"
#include "gpu/add_op.cuh"
#include "gpu/com_matrix_mul_op.cuh"
#include "gpu/stencil_1d_op.cuh"
#include "gpu/vec_add_op.cuh"
#include <functional>

namespace mop{
    // ------------ 向量加法 ------------
    // 泛型 AddKernel 定义，使用模板来支持不同的数据类型
    template<typename T>
    using add_op = std::function<void(T* in1, T* in2, T* out, const int size)>;

    // 根据设备类型返回不同的加法内核函数指针，支持泛型
    template<typename T>
    add_op<T> get_vec_add_op(mbase::DeviceType device_type);

    template <typename T>
    add_op<T> get_vec_add_op(mbase::DeviceType device_type) {
        if (device_type == mbase::DeviceType::HOST) {
            return nullptr;
        } else if (device_type == mbase::DeviceType::Device) {
            return [](T* in1, T* in2, T* out, const int size) {
                cudaop::vec_add_op_cu(in1, in2, out, size);
            };
        } else {
            LOG(FATAL) << "Unknown device type for get a add operator.";
            return nullptr;
        }
    }
    // ------------ END ------------

    // ------------ 矩阵乘法 ------------
    template <typename T>
    using com_matrix_mul_op = std::function<void(T* A, T* B, T* C, int row, int col)>;

    template <typename T>
    com_matrix_mul_op<T> get_com_matrix_mul_op(mbase::DeviceType device_type);

    template <typename T>
    com_matrix_mul_op<T> get_com_matrix_mul_op(mbase::DeviceType device_type) {
        if (device_type == mbase::DeviceType::HOST) {
            return nullptr;
        } else if (device_type == mbase::DeviceType::Device) {
            return [](const T* A, const T* B, T* C, int row, int col) {
                cudaop::com_matrix_mul_op_cu(A, B, C, row, col);
            };
        } else {
            LOG(FATAL) << "Unknown device type for get a com_matrix_mul operator.";
            return nullptr;
        }
    }
    // ------------ END ------------

    typedef void (*stencil_1d_op)(int *in, int *out, int arraySize, int padding);

    // 根据设备类型返回不同的加法内核
    stencil_1d_op get_stencil_1d_op(mbase::DeviceType device_type);
}

#endif