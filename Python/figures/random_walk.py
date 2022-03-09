import os

import numpy as np

from Python.Nano import IntVector
from Python.wrap import int_vec_to_mat
import matplotlib.pyplot as plt
import Python.Nano as Nano


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
    dim.x, dim.y, dim.z = (201, 1, 1)

    lattice = Nano.Lattice(dim)
    set_lattice(lattice)

    replicate_count = 64

    metropolis_list = []
    lattice_list = []
    gen_list = []

    for i in range(0, replicate_count):
        gen = Nano.RandomGenerator(123)

        values = np.zeros(dim.x, dtype=int)
        values[100] = 1

        array = IntVector(values.tolist())
        lattice.array_init(array)

        sim_params = Nano.SimulationParams()
        sim_params.default_events()
        sim = Nano.Simulation(sim_params, lattice, gen)

    occ_est = Nano.OccupancyEstimate(dim, lattice.energy_map)
    occ_est.run_KMC(sim, lattice, int(1e5))

    A_counts = occ_est.get_count_raw(1)

    plt.stem(A_counts)
    plt.show()

if __name__ == '__main__':
    main()