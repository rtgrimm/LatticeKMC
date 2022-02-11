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

        size_t get_type_count() const {
            return _types.size();
        }

        ParticleType get_particle_type(int32_t index) {
            return _types[index];
        }

        void add_particle_type(ParticleType id) {
            _types.push_back(id);
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
    };

    struct RandomGenerator {
        std::mt19937 gen;

        explicit RandomGenerator(size_t seed) : gen(seed) {}
    };

    class Lattice {
        struct ParticleData {
            Particle particle;
            IVec3D location;
        };

    public:
        const IVec3D size;
        EnergyMap energy_map;

        explicit Lattice(IVec3D size_) : size(size_),
                                         _particle_map(size_, Particle::invalid_particle) {}

        Particle& particle_at(IVec3D loc) {
            auto index = _particle_map.get(loc);
            return _particles[index].particle;
        }

        double site_energy(IVec3D center, ParticleType type) {
            auto interaction = [&] (IVec3D other_loc)  {
                auto other_type = particle_at(other_loc).type;

                auto J = energy_map.get_interaction(type, other_type);
                return J;
            };

            double energy = energy_map.get_field(type);

            for(auto& target : nearest(center)) {
                energy += interaction(target);
            }

            return energy;
        }

        template<class F>
        void init(F f) {
            int32_t particle_id = 0;

            for_all(size,[&] (IVec3D loc) {
                auto type = f(loc);

                ParticleData data {
                        Particle {type}, loc
                };

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

            for(auto& particle : _particles) {
                types.push_back(particle.particle.type);
            }

            return types;
        }

    private:
        std::vector<ParticleData> _particles;
        Tensor<int32_t> _particle_map;
    };



}