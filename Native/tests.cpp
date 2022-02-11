#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "lattice.hpp"

TEST_CASE("get and set for tensor", "[tensor]") {
    Nano::Tensor<float, true> test(Nano::IVec3D {
            5,
            12,
            2
    }, 0.0);

    SECTION("without periodic") {
        auto location = Nano::IVec3D {0, 1, 2};
        test.set(location, 5.0);
        REQUIRE(test.get(location) == 5.0);
    }

    SECTION("with periodic") {
        test.set(Nano::IVec3D {0, 1, 1}, 7.0);
        REQUIRE(test.get(Nano::IVec3D {5, 13, 3}) == 7.0);
    }
}

TEST_CASE("uniform init for lattice", "[lattice]") {\
    Nano::Lattice lattice(Nano::IVec3D {100, 100, 100});

    lattice.energy_map.set_uniform_binary(1.0, 1.0);

    Nano::RandomGenerator generator (123);
    lattice.uniform_init(generator);

    SECTION("check for uninitialized cells") {
        auto uninit_found = false;

        Nano::for_all(lattice.size, [&] (Nano::IVec3D loc) {
            if(lattice.particle_at(loc).type == Nano::Particle::invalid_particle) {
                uninit_found = true;
            }
        });

        REQUIRE(uninit_found == false);
    }

    SECTION("check for uniform distribution") {
        auto A_count = 0;
        auto B_count = 0;
        auto total = 0;

        Nano::for_all(lattice.size, [&] (Nano::IVec3D loc) {
            if(lattice.particle_at(loc).type == 1) {
                A_count++;
            } else {
                B_count++;
            }

            total++;
        });

        double A_frac = static_cast<float>(A_count) / static_cast<float>(total);
        double B_frac = static_cast<float>(B_count) / static_cast<float>(total);

        double diff = std::abs(A_frac - B_frac);

        REQUIRE(diff <= 1e-2);
    }
}