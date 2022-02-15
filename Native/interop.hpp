#pragma once
#include <cstdint>
#include <vector>

#include "base.hpp"

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

    VecData<int32_t> vector_data(std::vector<IVec3D> &target) {
        return {
                reinterpret_cast<int32_t*>(target.data()),
                target.size() * 3
        };
    }
}