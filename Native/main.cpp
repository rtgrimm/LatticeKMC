#include <iostream>
#include <string>
#include "lattice.hpp"
#include "catch.hpp"

#include "kmc.hpp"
#include "metropolis.hpp"

union FloatKey {
    double D;
    int64_t I;
};

int main() {
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




    for (int i = 0; i < 10000; ++i) {
        simulation.step();
    }

    return 0;
}
