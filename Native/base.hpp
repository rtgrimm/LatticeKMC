#pragma once

#include <cstdint>
#include <array>
#include <vector>

namespace Nano {
    struct IVec3D {
        int32_t x = 0;
        int32_t y = 0;
        int32_t z = 0;

        static constexpr int32_t dim = 3;

        int32_t V() const {
            return x * y * z;
        }

        bool operator==(const IVec3D &rhs) const {
            return x == rhs.x &&
                   y == rhs.y &&
                   z == rhs.z;
        }

        bool operator!=(const IVec3D &rhs) const {
            return !(rhs == *this);
        }

        IVec3D operator+(const IVec3D& other) const {
            return IVec3D {
              x + other.x,
              y + other.y,
              z + other.z,
            };
        }

        IVec3D operator-() const {
            return IVec3D {
                -x, -y, -z
            };
        }

        IVec3D operator-(const IVec3D& other) const {
            return *this + (-other);
        }

        double length_sq() const {
            auto x_f = static_cast<double>(x);
            auto y_f = static_cast<double>(y);
            auto z_f = static_cast<double>(z);

            return x_f*x_f + y_f*y_f + z_f*z_f;
        }

        IVec3D wrap(IVec3D size) const {
            auto wrap = [&] (int32_t index, int32_t bound) {
                auto rem = index % bound;

                if(rem < 0)
                    return bound + rem;

                return rem;
            };

            auto x_ = wrap(x, size.x);
            auto y_ = wrap(y, size.y);
            auto z_ = wrap(z, size.z);

            return IVec3D {x_, y_, z_};
        }
    };

    template<class F>
    void for_all(IVec3D vec, F f) {
        for (int i = 0; i < vec.x; ++i) {
            for (int j = 0; j < vec.y; ++j) {
                for (int k = 0; k < vec.z; ++k) {
                    IVec3D loc {i, j, k};
                    f(loc);
                }
            }
        }
    }

    template<class F>
    void nearest_neighbours(IVec3D loc, F f, size_t dim) {
        static constexpr std::array<IVec3D, 6> offsets = {
                IVec3D {-1, 0, 0},
                IVec3D {1, 0, 0},
                IVec3D {0, -1, 0},
                IVec3D {0, 1, 0},
                IVec3D {0, 0, -1},
                IVec3D {0, 0, 1},
        };

        for (int i = 0; i < dim * 2; ++i) {
            f(loc + offsets[i]);
        }
    }




    template<class T, bool Periodic = true>
    class Tensor {
    public:
        const IVec3D size;

        explicit Tensor(IVec3D size, T start_value = T()) : size(size) {
            _items.resize(size.V());
            std::fill(std::begin(_items), std::end(_items), start_value);
        }

        T& get(IVec3D location) {
            return _items[get_index(location)];
        }

        void set(IVec3D location, T value) {
            _items[get_index(location)] = value;
        }

        std::vector<T>& raw() {
            return _items;
        }

    private:
        int32_t get_index(IVec3D location) {
            if constexpr(Periodic) {
                location = location.wrap(size);
            }

            return location.x + location.y * size.x + location.z * size.x * size.y;
        }

        std::vector<T> _items;
    };



}