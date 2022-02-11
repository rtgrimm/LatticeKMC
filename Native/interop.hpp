#pragma once
#include <cstdint>
#include <vector>

namespace Nano::Interop {
    struct VecData {
        int32_t* ptr;
        size_t size;
    };


    VecData vector_data(std::vector<int32_t> &target) {
        return {
                target.data(),
                target.size()
        };
    }
}