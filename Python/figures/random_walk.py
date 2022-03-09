import os

import numpy as np

from Python.Nano import IntVector
from Python.wrap import int_vec_to_mat, double_vec_to_mat
import matplotlib.pyplot as plt
import Python.Nano as Nano
import deepdish as dd

def set_lattice(lattice):

    T = 0.3


    lattice.set_dim(Nano.LatticeDim_One)

    lattice.energy_map.add_particle_type(0)
    lattice.energy_map.add_particle_type(1)

    lattice.energy_map.set_interaction(1, 0, 0.0)

    lattice.energy_map.set_interaction(1, 1, 0.0)
    lattice.energy_map.set_interaction(0, 0, 0.0)

    lattice.energy_map.set_field(1, 0.0)
    lattice.energy_map.set_field(0, 0.0)

    lattice.energy_map.beta = 1 / T
    lattice.energy_map.allow_no_effect_move = True

def main():
    dim = Nano.IVec3D()
    dim.x, dim.y, dim.z = (300, 1, 1)

    bin_count = 100

    occ_est = Nano.OccupancyEstimate(300, 300 / bin_count, dim, 1)

    while True:
        replicate_count = 8

        sim_list = []
        lattice_list = []
        gen_list = []

        for i in range(0, replicate_count):
            lattice = Nano.Lattice(dim)
            set_lattice(lattice)

            gen = Nano.RandomGenerator(i)

            values = np.zeros(dim.x, dtype=int)
            values[150] = 1

            array = IntVector(values.tolist())
            lattice.array_init(array)

            #lattice.uniform_init(gen)
            lattice.enable_grid_tracking()

            sim_params = Nano.SimulationParams()
            sim_params.default_events()
            sim = Nano.Simulation(sim_params, lattice, gen)

            sim_list.append(sim)
            lattice_list.append(lattice)
            gen_list.append(gen)



        sim_list_ = Nano.SimulationVector(sim_list)



        Nano.run_kmc_parallel(sim_list_, int(1e5))

        for lattice in lattice_list:
            occ_est.add_particles(lattice)
            lattice.clear_history()

        bins = double_vec_to_mat(occ_est.get_bins_raw())
        times = double_vec_to_mat(occ_est.get_times())

        print(np.max([sim.get_time() for sim in sim_list]))

        bin_count = len(times)
        bin_grid = np.reshape(bins, newshape=(bin_count, dim.x))

        dd.io.save("random_walk_data", {
            "times" : times,
            "bin_grid" : bin_grid
        })







if __name__ == '__main__':
    main()