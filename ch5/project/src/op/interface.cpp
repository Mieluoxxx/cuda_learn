#include "interface.h"

#include <gpu/com_matrix_mul_op.cuh>
#include <gpu/stencil_1d_op.cuh>

namespace mop {
    stencil_1d_op get_stencil_1d_op(mbase::DeviceType device_type) {
        if (device_type == mbase::DeviceType::HOST) {
            return nullptr;
        } else if (device_type == mbase::DeviceType::Device) {
            return cudaop::stencil_1d_op_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a common_matrix_mul operator.";
            return nullptr;
        }
    }
}
