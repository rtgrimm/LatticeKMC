#pragma once

#include <execution>
#include <algorithm>
#include <utility>
#include "kmc.hpp"
#include "metropolis.hpp"

namespace Nano::Parallel {
    template<class S>
    void run_parallel(std::vector<S*> simulations, size_t step_count) {
        auto f = [&] (S* simulation) {
            simulation->step(step_count);
        };

        std::for_each(std::execution::par_unseq,
                      std::begin(simulations),
                      std::end(simulations), f);
    }

    void run_kmc_parallel(std::vector<KMC::Simulation*> simulations, size_t step_count) {
        run_parallel(std::move(simulations), step_count);
    }

    void run_metropolis_parallel(std::vector<Metropolis::Metropolis*> simulations, size_t step_count) {
        run_parallel(std::move(simulations), step_count);
    }
}