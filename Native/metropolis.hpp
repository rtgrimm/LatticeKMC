#pragma once

#include <iostream>
#include <cstdint>
#include <cassert>
#include <vector>
#include <limits>
#include <random>
#include <array>
#include <tuple>

#include "lattice.hpp"

namespace Nano::Metropolis {
    class Metropolis {
    public:
        Metropolis(Lattice* lattice_, RandomGenerator* rand_gen_) :
                x_random(0, lattice_->size.x - 1),
                y_random(0, lattice_->size.y - 1),
                z_random(0, lattice_->size.z - 1),
                lattice(lattice_), randGen(rand_gen_) {}


        Lattice& get_lattice() {
            return *lattice;
        }


        double acceptance_rate() const {
            return static_cast<double>(accepted_steps) / static_cast<double>(total_steps);
        }


        void step(size_t steps = 1) {
            for (int step = 0; step < steps; ++step) {
                lattice->save_state(static_cast<double>(total_steps));
                static_step();
            }
        }

    private:
        void static_step() {
            total_steps++;

            auto center = IVec3D {x_random(randGen->gen),
                                  y_random(randGen->gen),
                                  z_random(randGen->gen)};

            auto target = lattice->particle_at(center);

            auto new_state = gen_state();

            auto delta_E = lattice->site_energy(center, new_state)
                           - lattice->site_energy(center, target.type);

            if(!accept(delta_E))
                return;

            target.type = new_state;
            lattice->particle_at(center) = target;

            accepted_steps++;
        }



        Lattice* lattice;


        int32_t accepted_steps = 0;
        int32_t total_steps = 0;

        RandomGenerator* randGen;
        std::uniform_int_distribution<int32_t> x_random;
        std::uniform_int_distribution<int32_t> y_random;
        std::uniform_int_distribution<int32_t> z_random;

        std::uniform_real_distribution<double> real_random;


        bool accept(double delta_E) {
            if(delta_E <= 0) {
                return true;
            }

            auto r = real_random(randGen->gen);
            return r <= std::exp(-lattice->energy_map.beta * delta_E);
        }

        int32_t gen_state() {
            std::uniform_int_distribution<int32_t> state_random (
                    0, static_cast<int32_t>(lattice->energy_map.get_type_count()) - 1);

            return lattice->energy_map.get_particle_type(state_random(randGen->gen));
        }
    };
}