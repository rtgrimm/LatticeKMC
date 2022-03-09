import os

from wrap import double_vec_to_mat
from wrap import int_vec_to_mat
import matplotlib.pyplot as plt
import Nano
import numpy as np

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

def main_2():
    dim = Nano.IVec3D()
    dim.x, dim.y, dim.z = (201, 1, 1)



    replicate_count = 1

    sim_list = []
    lattice_list = []
    gen_list = []

    for i in range(0, replicate_count):
        lattice = Nano.Lattice(dim)
        set_lattice(lattice)

        gen = Nano.RandomGenerator(i)

        values = np.zeros(dim.x, dtype=int)
        values[100] = 1

        #array = IntVector(values.tolist())
        #lattice.array_init(array)

        lattice.uniform_init(gen)
        lattice.enable_grid_tracking()

        sim_params = Nano.SimulationParams()
        sim_params.default_events()
        sim = Nano.Simulation(sim_params, lattice, gen)

        sim_list.append(sim)
        lattice_list.append(lattice)
        gen_list.append(gen)



    sim_list_ = Nano.SimulationVector(sim_list)
    Nano.run_kmc_parallel(sim_list_, int(1e4))

    max_time = np.max([sim.get_time() for sim in sim_list_])

    occ_est = Nano.OccupancyEstimate(max_time, max_time / 100.0, dim, 1)

    for lattice in lattice_list:
        occ_est.add_particles(lattice)

    bins = double_vec_to_mat(occ_est.get_bins_raw())
    times = double_vec_to_mat(occ_est.get_times())

    bin_count = len(times)

    bin_grid = np.reshape(bins, newshape=(bin_count, dim.x))

    print()

def main_1():
    dim = Nano.IVec3D()
    dim.x, dim.y, dim.z = (100, 100, 1)

    T = 0.3

    lattice = Nano.Lattice(dim)
    lattice.set_dim(Nano.LatticeDim_Two)

    lattice.energy_map.add_particle_type(-1)
    lattice.energy_map.add_particle_type(1)

    lattice.energy_map.set_interaction(1, -1, 1.0)

    lattice.energy_map.set_interaction(1, 1, -1.0)
    lattice.energy_map.set_interaction(-1, -1, -1.0)

    lattice.energy_map.set_field(1, 0.0)
    lattice.energy_map.set_field(-1, 0.0)

    lattice.energy_map.beta = 1 / T
    lattice.energy_map.allow_no_effect_move = True

    gen = Nano.RandomGenerator(123)
    lattice.uniform_init(gen)

    metropolis = Nano.Metropolis(lattice, gen)
    #metropolis.step(int(1e7))

    sim_params = Nano.SimulationParams()
    sim_params.default_events()
    sim = Nano.Simulation(sim_params, lattice, gen)

    plt.ion()

    energy_list = []
    mag_list = []
    time_list = []

    while True:

        output = int_vec_to_mat(lattice.get_types())
        grid = output.reshape((dim.x, dim.y, dim.z))

        energy_list.append(lattice.total_energy())
        mag_list.append(lattice.mean_magnetization())
        time_list.append(sim.get_time())

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(grid)

        plt.subplot(1, 2, 2)
        plt.plot(energy_list)
        plt.gca().twinx().plot(mag_list)

        sim.step(int(1e2))
        plt.draw()
        plt.pause(1e-9)

if __name__ == '__main__':
    main_2()