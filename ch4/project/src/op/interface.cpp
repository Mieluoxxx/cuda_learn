#include "interface.h"

#include <gpu/com_matrix_mul_op.cuh>
#include <gpu/stencil_1d_op.cuh>

namespace mop {
    AddKernel get_add_op(mbase::DeviceType device_type) {
        if (device_type == mbase::DeviceType::HOST) {
            // 如果是 HOST，返回 nullptr，意味着没有使用 GPU
            return nullptr;
        } else if (device_type==mbase::DeviceType::Device) {
            return cudaop::add_op_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a add kernel.";
            return nullptr;
        }
    }

    com_matrix_mul_op get_com_matrix_mul_op(mbase::DeviceType device_type) {
        if (device_type == mbase::DeviceType::HOST) {
            // 如果是 HOST，返回 nullptr，意味着没有使用 GPU
            return nullptr;
        } else if (device_type==mbase::DeviceType::Device) {
            return cudaop::com_matrix_mul_op_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a add kernel.";
            return nullptr;
        }
    }

    stencil_1d_op get_stencil_1d_op(mbase::DeviceType device_type) {
        if (device_type == mbase::DeviceType::HOST) {
            // 如果是 HOST，返回 nullptr，意味着没有使用 GPU
            return nullptr;
        } else if (device_type==mbase::DeviceType::Device) {
            return cudaop::stencil_1d_op_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a add kernel.";
            return nullptr;
        }
    }


}
