#pragma once

#include "lattice.hpp"

#include <map>
#include <unordered_map>
#include <utility>
#include <set>

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
    };

    struct Event {
        double rate = 0.0;
        Events::Type type = Events::Type::Nothing;
        IVec3D center;
        IVec3D target;
    };

    class EventCache {
    public:
        using GroupId = double;
        using EventId = int32_t;
        using Key = std::tuple<GroupId, EventId>;

        explicit EventCache(IVec3D size) : _id_map(size) {}

        bool empty() const {
            return _group_map.empty();
        }

        void add_event(const Event& event) {
            auto& event_list = _group_map[event.rate];
            auto key = std::make_tuple(
                    event.rate, static_cast<int32_t>(event_list.size()));

            _group_map[event.rate].push_back(event);
            _id_map.get(event.center).push_back(key);
        }

        void clear_location(IVec3D center) {
            auto& elements = _id_map.get(center);

            for(auto& key : elements) {
                erase_from_group_map(key);
            }

            elements.clear();
        }

        std::tuple<Event, double> choose_event(RandomGenerator& generator) {
            double rate_total = 0.0;
            
            for(auto& [rate, list] : _group_map) {
                rate_total += rate * static_cast<double>(list.size());
            }

            double partial_sum = 0.0;
            auto r_1 = uniform_real(generator.gen) * rate_total;
            
            std::vector<Event>* event_list = &(--std::end(_group_map))->second;
            
            for(auto& [rate, list] : _group_map) {
                partial_sum += rate * static_cast<double>(list.size());

                if(partial_sum > r_1) {
                    event_list = &list;
                    break;
                }
            }

            std::uniform_int_distribution<size_t> uniform_index(0, event_list->size() - 1);
            
            auto index = uniform_index(generator.gen);
            auto event = (*event_list)[index];

            return std::make_tuple(event, rate_total);
        }

    private:
        std::uniform_real_distribution<double> uniform_real =
                std::uniform_real_distribution<double>(0.0, 1.0);
        
        void erase_from_group_map(Key key) {
            auto& event_list = _group_map[std::get<0>(key)];
            event_list[std::get<1>(key)] = *(std::end(event_list) - 1);
            event_list.pop_back();
        }

        Tensor<std::vector<std::tuple<GroupId, EventId>>> _id_map;
        std::map<GroupId, std::vector<Event>> _group_map;
    };

    class Simulation {
        using VecKey = std::tuple<int32_t, int32_t, int32_t>;
        using LocationEventMap = std::multimap<VecKey, Event>;

    public:
        SimulationParams params;
        Lattice* lattice;
        RandomGenerator* random_generator;

        Simulation(SimulationParams params_, Lattice *lattice, RandomGenerator *randomGenerator)
        : params(std::move(params_)), lattice(lattice), random_generator(randomGenerator), _cache(lattice->size) {}

        /*std::vector<double> get_event_rates() {
            std::vector<double> rates;

            for(auto& pair : _location_event_map) {
                auto rate = pair.second.rate;
                rates.push_back(rate);
            }

            return rates;
        }*/

        void step() {
            if(_cache.empty()) {
                init_events();
            }

            auto [selected_event, k_total] = _cache.choose_event(*random_generator);

            Events::Dispatch::execute(selected_event.type, *lattice,
                            selected_event.center, selected_event.target);

            update_surrounding_events(
                    selected_event.center, selected_event.target);

            update_surrounding_events(
                    selected_event.target, selected_event.center);

            auto r_2 = uniform_real(random_generator->gen);
            auto delta_t = -std::log(r_2) / k_total;

            _time += delta_t;
        }

        void multi_step(size_t count) {
            for (int i = 0; i < count; ++i) {
                step();
            }
        }

        double get_time() const {
            return _time;
        }

    private:
        EventCache _cache;

        double _time = 0.0;

        std::uniform_real_distribution<double> uniform_real =
                std::uniform_real_distribution<double>(0.0, 1.0);



        void init_events() {
            for_all(lattice->size, [&] (IVec3D loc) {
                update_events(loc, {});
            });
        }

        void update_surrounding_events(IVec3D center, std::optional<IVec3D> ignore) {
            update_events(center, ignore);

            for(auto target : nearest(center)) {
                update_events(target, ignore);
            }
        }

        void update_events(IVec3D loc, std::optional<IVec3D> ignore) {
            _cache.clear_location(loc);

            for(auto& event_type : params.allowed_events) {
                for(auto target : nearest(loc)) {
                    if(ignore.has_value() && target == ignore.value()) {
                        continue;
                    }

                    auto rate = Events::Dispatch::calc_rate(
                            event_type, *lattice, loc, target);

                    if(rate == 0.0) {
                        continue;
                    }

                    Event event {
                        rate, event_type, loc, target
                    };

                    _cache.add_event(event);
                }

            }
        }
    };
}