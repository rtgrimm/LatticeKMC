from matplotlib import pyplot as plt

from Python import Nano
from Python.Nano import vector_data, MetropolisVector, SimulationVector, Simulation
from Python.demos import set_model
from Python.style import set_style, pal
from Python.wrap import int_vec_to_mat, double_vec_to_mat, IVec3D_to_mat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import Python.dep_graph as dg


def run_multiplex_MSD(config_model):
    dim = Nano.IVec3D()
    dim.x, dim.y, dim.z = (250, 250, 1)

    replicate_count = 48

    metropolis_list = []
    kmc_list = []
    lattice_list = []
    gen_list = []

    for i in range(0, replicate_count):
        lattice = Nano.Lattice(dim)
        lattice.set_dim(Nano.LatticeDim_Two)
        lattice.enable_particle_tracking()

        config_model(lattice)

        gen = Nano.RandomGenerator(i)
        lattice.uniform_init(gen)

        metropolis = Nano.Metropolis(lattice, gen)
        metropolis_list.append(metropolis)
        lattice_list.append(lattice)
        gen_list.append(gen)

        sim_params = Nano.SimulationParams()
        sim_params.default_events()
        sim = Nano.Simulation(sim_params, lattice, gen)
        kmc_list.append(sim)

    Nano.run_metropolis_parallel(MetropolisVector(metropolis_list), int(1e7))
    Nano.run_kmc_parallel(SimulationVector(kmc_list), int(1e6))

    max_time = np.max([sim.get_time() for sim in kmc_list])
    bin_width = 1e-1
    msd_estimate = Nano.MSDEstimate(max_time, bin_width)

    for lattice, sim in zip(lattice_list, kmc_list):
        msd_estimate.bin_displacements(lattice, 1)


    series = msd_estimate.get_MSD_series()

    times = double_vec_to_mat(series.times)
    means = double_vec_to_mat(series.values)

    return times, means




def MSD():

    time_list = []
    MSD_list = []
    param_list = []

    model_configs = []

    for mu_v in np.linspace(0.1, 0.5, 5):
        model_configs.append({"mu_v" : mu_v})


    for config in model_configs:
        print(config)

        times, means = run_multiplex_MSD(
            lambda lattice: set_model(
                lattice, 2.0, 2.0, mu_v = config["mu_v"], T = 0.55)
        )

        time_list.append(times)
        MSD_list.append(means)
        param_list.append(config)

    return time_list, MSD_list, param_list


def plot_MSD(MSD):
    plt.figure(figsize=(32, 18))

    time_list, MSD_list, param_list = MSD

    for times, means, params in zip(time_list, MSD_list, param_list):
        #plt.scatter(times, means, s=10.0)
        mu_v = params["mu_v"]
        plt.plot(times, means, label = f"$\mu_v = {mu_v}$")

    plt.legend(frameon=False)
    plt.ylabel("MSD")
    plt.xlabel("Time")
    plt.savefig("figures/MSD_mu_v.pdf")

def main():
    set_style()

    graph = dg.Graph(dg.HDF5("data_cache"))

    graph.register_all(map(dg.task, [
        plot_MSD,
        MSD
    ]))

    graph("plot_MSD")

if __name__ == '__main__':
    main()



