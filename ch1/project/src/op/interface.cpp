#include "interface.h"

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

}