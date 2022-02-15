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




    for (int i = 0; i < 10000; ++i) {
        simulation.step();
    }

}

struct Event {
    double rate = 0.0;
    Nano::KMC::Events::Type type = Nano::KMC::Events::Type::Nothing;
    Nano::IVec3D center;
    Nano::IVec3D target;
};

template<class F>
long clock_f(F f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int main() {
    auto length = static_cast<int>(1e6);

    if(false) {
        std::set<int> events;



        for (int i = 0; i < length; ++i) {
            events.insert(i);
        }

        std::cout << clock_f([&] () {
            auto it = std::begin(events);
            std::advance(it,length / 2);

            std::cout << *it << std::endl;
        }) << std::endl;
    }

    std::mt19937 gen(123);

    std::uniform_int_distribution dist(0, length);

    std::cout << clock_f([&]  {
        dist(gen);
    });

    return 0;
}
