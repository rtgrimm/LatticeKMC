#pragma once
#include <vector>
#include <map>
#include <random>

#include "base.hpp"

namespace Nano {
    using ParticleId = int32_t;
    using ParticleType = int32_t;

    class EnergyMap {
    public:
        double beta = 1.0;
        bool allow_no_effect_move = false;

        size_t get_type_count() const {
            return _types.size();
        }

        ParticleType get_particle_type(int32_t index) {
            return _types[index];
        }

        void add_particle_type(ParticleType id) {
            _types.push_back(id);
        }

        void reset() {
            _interactions.clear();
            _field.clear();
            _types.clear();
        }

        void set_uniform_binary(double interaction, double field) {
            auto b = -1;
            auto a = 1;

            add_particle_type(a);
            add_particle_type(b);

            set_interaction(b, a, -interaction);
            set_interaction(b, b, interaction);
            set_interaction(a, a, interaction);
            set_field(b, -field);
            set_field(a, field);
        }

        void set_interaction(signed int i, signed int j, double J) {
            _interactions[std::make_tuple(i, j)] = J;
            _interactions[std::make_tuple(j, i)] = J;
        }

        double get_interaction(signed int i, signed int j)  {
            return _interactions[std::make_tuple(i, j)];
        }

        double get_field(signed int i) {
            return _field[i];
        }

        void set_field(signed int i, double B) {
            _field[i] = B;
        }

    private:
        std::map<std::tuple<ParticleType, ParticleType>, double> _interactions;
        std::map<ParticleType, double> _field;
        std::vector<ParticleType> _types;
    };

    struct Particle {
        static constexpr int32_t invalid_particle = -404;
        ParticleType type;
        IVec3D start_pos;
        std::vector<IVec3D> hop_list;
        std::vector<double> time_list;
    };

    struct RandomGenerator {
        std::mt19937 gen;

        explicit RandomGenerator(size_t seed) : gen(seed) {}
    };

    enum class LatticeDim {
        One, Two, Three
    };

    class Lattice {

    public:
        const IVec3D size;
        EnergyMap energy_map;

        void enable_particle_tracking() {
            _particle_tracking_enabled = true;
        }

        void disable_particle_tracking() {
            _particle_tracking_enabled = false;
        }

        void set_dim(LatticeDim dim) {
            _dim = dim;
        }

        explicit Lattice(IVec3D size_) :
            size(size_), _particle_map(size_, Particle::invalid_particle) {}

        Particle& particle_at(IVec3D loc) {
            auto index = _particle_map.get(loc);
            return _particles[index];
        }

        void swap(IVec3D from_loc, IVec3D to_loc, double time) {
            auto from_index = _particle_map.get(from_loc);
            auto to_index = _particle_map.get(to_loc);

            if(_particle_tracking_enabled) {
                auto from_to_hop = to_loc - from_loc;
                auto to_from_hop = -from_to_hop;

                _particles[from_index].hop_list.push_back(from_to_hop);
                _particles[to_index].hop_list.push_back(to_from_hop);

                _particles[from_index].time_list.push_back(time);
                _particles[to_index].time_list.push_back(time);
            }

            _particle_map.set(from_loc, to_index);
            _particle_map.set(to_loc, from_index);
        }


        double site_energy(IVec3D center, ParticleType type) {
            auto interaction = [&] (IVec3D other_loc)  {
                auto other_type = particle_at(other_loc).type;

                auto J = energy_map.get_interaction(type, other_type);
                return J;
            };

            double energy = energy_map.get_field(type);

            nearest(center, [&] (IVec3D target) {
                energy += interaction(target);
            });

            return energy;
        }

        double total_energy() {
            double total = 0.0;

            for_all(size,[&] (IVec3D loc) {
                total += site_energy(loc, particle_at(loc).type);
            });

            return total;
        }

        double mean_magnetization() {
            double total = 0.0;

            for_all(size,[&] (IVec3D loc) {
                total += static_cast<double>(particle_at(loc).type);
            });

            return total / size.V();
        }

        template<class F>
        void nearest(IVec3D center, F f) {
            if(_dim == LatticeDim::One) {
                nearest_neighbours(center, f, 1);
            } else if(_dim == LatticeDim::Two) {
                nearest_neighbours(center, f, 2);
            } else if(_dim == LatticeDim::Three) {
                nearest_neighbours(center, f, 3);
            }
        }

        template<class F>
        void init(F f) {
            int32_t particle_id = 0;

            for_all(size,[&] (IVec3D loc) {
                auto type = f(loc);

                Particle data {type, loc};

                _particle_map.set(loc, particle_id);
                _particles.push_back(data);
                particle_id++;
            });
        }

        void uniform_init(RandomGenerator& random_generator) {
            init([&] (IVec3D loc) {
                std::uniform_int_distribution<int32_t> type_random (
                        0, static_cast<int32_t>(energy_map.get_type_count()) - 1);

                auto index = type_random(random_generator.gen);
                auto type = energy_map.get_particle_type(index);

                return type;
            });
        }

        std::vector<int32_t> get_types() {
            std::vector<int32_t> types;

            for_all(_particle_map.size, [&] (IVec3D& loc) {
                types.push_back(particle_at(loc).type);
            });

            return types;
        }

        const std::vector<Particle>& get_particles() const {
            return _particles;
        }

    private:
        bool _particle_tracking_enabled = false;
        std::vector<Particle> _particles;
        Tensor<int32_t> _particle_map;
        LatticeDim _dim = LatticeDim::Three;
    };
}