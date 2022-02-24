#pragma once

#include "lattice.hpp"

#include <map>
#include <unordered_map>
#include <utility>
#include <set>
#include <iostream>
#include <optional>

namespace Nano::KMC {
    namespace Events {
        enum class Type {
            Swap,
            Nothing
        };

        class Dispatch {
            struct Swap {
                static void execute(Lattice& lattice, IVec3D from_loc, IVec3D to_loc, double time) {
                    lattice.swap(from_loc, to_loc, time);
                }

                static double rate(Lattice& lattice, IVec3D from_loc, IVec3D to_loc, double time) {
                    auto from = lattice.particle_at(from_loc);
                    auto to = lattice.particle_at(to_loc);

                    if(from.type == to.type) {
                        return 0.0;
                    }

                    auto E_i = lattice.site_energy(from_loc, from.type)
                               + lattice.site_energy(to_loc, to.type);

                    auto E_f = lattice.site_energy(from_loc, to.type)
                               + lattice.site_energy(to_loc, from.type);

                    auto delta_E = E_f - E_i;

                    if(delta_E > 0) {
                        return std::exp(-delta_E * lattice.energy_map.beta);
                    } else {
                        return 1.0;
                    }
                }
            };

        public:
            static void execute(Type type, Lattice& lattice, IVec3D center, IVec3D target, double time) {
                switch (type) {
                    case Type::Swap:
                        Swap::execute(lattice, center, target, time);
                        break;
                    case Type::Nothing:
                        break;
                }
            }

            static double calc_rate(Type type, Lattice& lattice, IVec3D center, IVec3D target, double time) {
                switch (type) {
                    case Type::Swap:
                        return Swap::rate(lattice, center, target, time);
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
            if(event.rate == 0.0) {
                return;
            }

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

    struct TimeBin {
        double t_0 = 0.0;
        double t_1 = 0.0;

        std::vector<double> displacement_bins;
    };

    struct TimeSeries {
        std::vector<double> times;
        std::vector<double> values;
    };

    class MSDEstimate {
    public:
        MSDEstimate(double maxTime, double timeBinWidth) : _max_time(maxTime), _time_bin_width(timeBinWidth) {
            build_time_bins();
        }

        void bin_displacements(Lattice* lattice, ParticleType type) {
            auto& particles = lattice->get_particles();

            for(auto& particle : particles) {
                if(particle.type != type)
                    continue;

                auto& time_list = particle.time_list;
                auto& hop_list = particle.hop_list;

                IVec3D position {0, 0, 0};

                for (int i = 0; i < time_list.size(); ++i) {
                    auto time = time_list[i];
                    auto hop = hop_list[i];
                    auto index = static_cast<int32_t>(floor(time * _time_bins.size() / _max_time));

                    auto displacement = position.length_sq();
                    _time_bins[index].displacement_bins.push_back(displacement);
                    position = position + hop;
                }
            }
        }


        TimeSeries get_MSD_series() {
            TimeSeries series;

            for (auto& time_bin : _time_bins) {
                auto& displacement_bin = time_bin.displacement_bins;

                if(displacement_bin.size() == 0) {
                    continue;
                }

                auto mean = 0.0;
                for(auto& x : displacement_bin) { mean += x;}
                
                mean /= displacement_bin.size();
                
                series.values.push_back(mean);
                series.times.push_back(time_bin.t_0);
            }

            return std::move(series);
        }

    private:
        double _max_time;
        double _time_bin_width;

        void build_time_bins() {
            auto time_bin_count = static_cast<int32_t>(ceil(_max_time / _time_bin_width));
            _time_bins.reserve(time_bin_count);

            for (int i = 0; i < time_bin_count; ++i) {
                _time_bins.push_back({
                    _time_bin_width * i, _time_bin_width * (i + 1)});
            }
        }

        std::vector<TimeBin> _time_bins;
    };

    class Simulation {
    public:
        SimulationParams params;
        Lattice* lattice;
        RandomGenerator* random_generator;

        Simulation(SimulationParams params_, Lattice *lattice, RandomGenerator *randomGenerator)
        : params(std::move(params_)), lattice(lattice), random_generator(randomGenerator), _cache(lattice->size) {}

        void step() {
            if(_cache.empty()) {
                init_events();
            }

            auto [selected_event, k_total] = _cache.choose_event(*random_generator);

            Events::Dispatch::execute(selected_event.type, *lattice,
                            selected_event.center, selected_event.target, _time);

            update_surrounding_events(
                    selected_event.center, selected_event.target);

            update_surrounding_events(
                    selected_event.target, selected_event.center);

            auto r_2 = uniform_real(random_generator->gen);
            auto delta_t = -std::log(r_2) / k_total;

            _time += delta_t;
        }

        void step(size_t count) {
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

            lattice->nearest(center, [&] (IVec3D target) {
                update_events(target, ignore);
            });
        }

        void update_events(IVec3D loc, std::optional<IVec3D> ignore) {
            _cache.clear_location(loc);

            for(auto& event_type : params.allowed_events) {
                lattice->nearest(loc, [&] (IVec3D target) {
                    if(ignore.has_value() && target == ignore.value()) {
                        return;
                    }

                    auto rate = Events::Dispatch::calc_rate(
                            event_type, *lattice, loc, target, _time);

                    if(rate == 0.0) {
                        return;
                    }

                    Event event {
                            rate, event_type, loc, target
                    };

                    _cache.add_event(event);
                });

            }
        }
    };
}