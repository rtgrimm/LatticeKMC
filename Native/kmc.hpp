#pragma once

#include "lattice.hpp"

#include <map>
#include <unordered_map>
#include <utility>

namespace Nano::KMC {
    namespace Events {
        enum class Type {
            Swap,
            Nothing
        };



        class Dispatch {
            struct Swap {



                static void execute(Lattice& lattice, IVec3D from_loc, IVec3D to_loc) {
                    auto from = lattice.particle_at(from_loc);
                    auto to = lattice.particle_at(to_loc);

                    lattice.particle_at(from_loc) = to;
                    lattice.particle_at(to_loc) = from;
                }

                static double rate(Lattice& lattice, IVec3D from_loc, IVec3D to_loc) {
                    auto from = lattice.particle_at(from_loc);
                    auto to = lattice.particle_at(to_loc);

                    auto E_i = lattice.site_energy(from_loc, from.type)
                               + lattice.site_energy(to_loc, to.type);

                    auto E_f = lattice.site_energy(from_loc, to.type)
                               + lattice.site_energy(to_loc, from.type);

                    auto delta_E = E_f - E_i;

                    auto rate = std::exp(-delta_E * lattice.energy_map.beta);

                    return rate;
                }
            };

        public:
            static void execute(Type type, Lattice& lattice, IVec3D center, IVec3D target) {
                switch (type) {
                    case Type::Swap:
                        Swap::execute(lattice, center, target);
                        break;
                    case Type::Nothing:
                        break;
                }
            }

            static double calc_rate(Type type, Lattice& lattice, IVec3D center, IVec3D target) {
                switch (type) {
                    case Type::Swap:
                        return Swap::rate(lattice, center, target);
                    default:
                        return 0;
                }
            }
        };
    }



    struct SimulationParams {
        std::vector<Events::Type> allowed_events;

        void default_events() {
            allowed_events.push_back(Events::Type::Swap);
        }
    } ;

    class Simulation {
        struct Event {
            double rate = 0.0;
            Events::Type type = Events::Type::Nothing;
            IVec3D center;
            IVec3D target;
        };

        using VecKey = std::tuple<int32_t, int32_t, int32_t>;
        using LocationEventMap = std::multimap<VecKey, Event>;



    public:
        SimulationParams params;
        Lattice* lattice;
        RandomGenerator* random_generator;

        Simulation(SimulationParams params_, Lattice *lattice, RandomGenerator *randomGenerator)
        : params(std::move(params_)), lattice(lattice), random_generator(randomGenerator) {}

        std::vector<double> get_event_rates() {
            std::vector<double> rates;

            for(auto& pair : _location_event_map) {
                auto rate = pair.second.rate;
                rates.push_back(rate);
            }

            return rates;
        }

        void step() {
            if(_location_event_map.empty()) {
                init_events();
            }

            std::vector<Event> events;
            double k_total = 0.0;

            for(auto& pair : _location_event_map) {
                auto event = pair.second;
                k_total += event.rate;

                events.push_back(event);
            }

            double partial_sum = 0.0;
            auto r_1 = uniform_real(random_generator->gen) * k_total;

            auto i = 0;

            for(auto& event : events) {
                partial_sum += event.rate;

                if(partial_sum > r_1) {
                    break;
                }

                i++;
            }

            auto& selected_event = events[i];

            Events::Dispatch::execute(selected_event.type, *lattice,
                            selected_event.center, selected_event.target);

            update_surrounding_events(selected_event.center);
            update_surrounding_events(selected_event.target);

            auto r_2 = uniform_real(random_generator->gen);
            auto delta_t = -std::log(r_2) / k_total;

            _time += delta_t;
        }

        double get_time() const {
            return _time;
        }

    private:
        double _time = 0.0;

        std::uniform_real_distribution<double> uniform_real =
                std::uniform_real_distribution<double>(0.0, 1.0);

        LocationEventMap _location_event_map;

        void init_events() {
            for_all(lattice->size, [&] (IVec3D loc) {
                update_events(loc);
            });
        }

        void update_surrounding_events(IVec3D center) {
            update_events(center);

            for(auto target : nearest(center)) {
                update_events(target);
            }
        }

        void update_events(IVec3D loc) {
            auto key = std::make_tuple(loc.x, loc.y, loc.z);
            _location_event_map.erase(key);

            for(auto& event_type : params.allowed_events) {
                for(auto target : nearest(loc)) {
                    auto rate = Events::Dispatch::calc_rate(
                            event_type, *lattice, loc, target);

                    if(rate == 0.0) {
                        continue;
                    }

                    Event event {
                        rate, event_type, loc, target
                    };

                    _location_event_map.insert({key, event});
                }

            }
        }
    };
}