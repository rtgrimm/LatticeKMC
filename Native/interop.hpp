#pragma once
#include <cstdint>
#include <vector>

namespace Nano::Interop {

    template<class T>
    struct VecData {
        T* ptr;
        size_t size;
    };


    VecData<int32_t> vector_data(std::vector<int32_t> &target) {
        return {
                target.data(),
                target.size()
        };
    }

    VecData<double> vector_data(std::vector<double> &target) {
        return {
                target.data(),
                target.size()
        };
    }
}