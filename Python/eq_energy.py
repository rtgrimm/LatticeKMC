import os
from wrap import int_vec_to_mat
import matplotlib.pyplot as plt
import Nano

def main_2():
    dim = Nano.IVec3D()
    dim.x, dim.y, dim.z = (100, 1, 1)

    T = 0.3

    lattice = Nano.Lattice(dim)
    lattice.set_dim(Nano.LatticeDim_One)

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

    occ_est = Nano.OccupancyEstimate(dim, lattice.energy_map)
    occ_est.run_KMC(sim, lattice, int(1e6))

    A_counts = occ_est.get_count_raw(1)

    plt.stem(A_counts)
    plt.show()

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