#include <iostream>
#include <string>
#include "lattice.hpp"
#include "catch.hpp"

#include "kmc.hpp"
#include "metropolis.hpp"

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
    Nano::Lattice lattice(Nano::IVec3D {10, 10, 10});



    lattice.energy_map.set_uniform_binary(1.0, 1.0);


    Nano::RandomGenerator generator (123);
    lattice.uniform_init(generator);

    Nano::Metropolis::Metropolis metropolis(&lattice, &generator);

    metropolis.step(1000);

    Nano::KMC::SimulationParams params;

    params.allowed_events.push_back(Nano::KMC::Events::Type::Swap);

    Nano::KMC::Simulation simulation(
            params, &lattice, &generator);


    auto step_count = 100000;

    //auto time = clock_f([&] {
        for (int i = 0; i < step_count; ++i) {
            simulation.step();
        }
    //}) / step_count;

    //std::cout << "Time:" << time << std::endl;

}


int main() {
    run_kmc();

    return 0;
}
