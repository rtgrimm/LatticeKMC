#include <iostream>
#include <string>
#include "lattice.hpp"
#include "catch.hpp"

#include "kmc.hpp"
#include "metropolis.hpp"

#include "parallel.hpp"

#include <set>
#include <chrono>
#include <random>

union FloatKey {
    double D;
    int64_t I;
};


template<class F>
long clock_f(F f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}


void run_kmc() {
    Nano::IVec3D dim {100,100,1};

    auto T = 0.3;

    auto lattice = Nano::Lattice(dim);
    lattice.set_dim(Nano::LatticeDim::Two);

    lattice.energy_map.add_particle_type(-1);
    lattice.energy_map.add_particle_type(1);

    lattice.energy_map.set_interaction(1, -1, 1.0);

    lattice.energy_map.set_interaction(1, 1, -1.0);
    lattice.energy_map.set_interaction(-1, -1, -1.0);

    lattice.energy_map.set_field(1, 0.0);
    lattice.energy_map.set_field(-1, 0.0);

    lattice.energy_map.beta = 1 / T;
    lattice.energy_map.allow_no_effect_move = false;

    Nano::RandomGenerator gen(123);
    lattice.uniform_init(gen);

    Nano::KMC::SimulationParams sim_params;
    sim_params.default_events();
    Nano::KMC::Simulation sim(sim_params, &lattice, &gen);

    for (int i = 0; i < 100; ++i) {
        sim.step(10000);
        std::cout << lattice.total_energy() << std::endl;
    }

}


int main() {
    run_kmc();

    return 0;
}
