from itertools import islice

from matplotlib import pyplot as plt

from Python import Nano
from Python.Nano import vector_data, MetropolisVector, SimulationVector, Simulation
from Python.demos import set_model, plot_point_cloud
from Python.style import set_style, pal
from Python.wrap import int_vec_to_mat, double_vec_to_mat, IVec3D_to_mat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import Python.dep_graph as dg


def run_metro(dim_, params_list):
    dim = Nano.IVec3D()
    dim.x, dim.y, dim.z = dim_

    metropolis_list = []
    lattice_list = []
    gen_list = []

    for params in params_list:
        lattice = Nano.Lattice(dim)

        if(dim.z == 1):
            lattice.set_dim(Nano.LatticeDim_Two)
        else:
            lattice.set_dim(Nano.LatticeDim_Three)

        set_model(lattice, 2.0, 2.0, params["mu_v"], params["T"], J_v_b=0.0, J_v_a=0.0)

        gen = Nano.RandomGenerator(123)
        lattice.uniform_init(gen)

        metropolis = Nano.Metropolis(lattice, gen)
        metropolis_list.append(metropolis)
        lattice_list.append(lattice)
        gen_list.append(gen)

    sim_list = MetropolisVector(metropolis_list)
    Nano.run_metropolis_parallel(sim_list, int(1e7))

    grid_list = []

    for lattice in lattice_list:
        output = int_vec_to_mat(lattice.get_types())

        if(dim.z == 1):
            grid = output.reshape((dim.x, dim.y))
        else:
            grid = output.reshape((dim.x, dim.y, dim.z))

        grid_list.append(grid)

    return grid_list, params_list

def build_params(T_count, mu_v_count):
    params_list = []

    for mu_v in np.linspace(0.05, 0.5, mu_v_count):
        for T in np.linspace(0.1, 1.0, T_count):
            params_list.append({
                "mu_v" : mu_v,
                "T" : T
            })

    return params_list


def metro_grid_2D():
    return run_metro((250, 250, 1), build_params(5, 5))


def metro_grid_3D():
    return run_metro((25, 25, 25), build_params(1, 3))


def plot_metro_2D(metro_grid_2D):
    N = len(metro_grid_2D[0])
    side_dim = np.ceil(np.sqrt(N))

    plt.figure(figsize=(32, 18))

    i = 1

    for grid, params in zip(*metro_grid_2D):
        T = params["T"]
        mu_v = params["mu_v"]


        plt.subplot(side_dim, side_dim, i)
        plt.title(f"$T = {T}$; $\mu_v={mu_v}$", fontdict = {'fontsize' : 20})
        plt.imshow(grid)
        plt.axis("off")

        i += 1

    plt.savefig("metro_grid.pdf")

def plot_metro_3D(metro_grid_3D):

    colors = list(islice(pal("bright", 3), 3))
    pairs = list(zip(*metro_grid_3D))

    N = len(pairs) * 3
    side_dim = np.ceil(np.sqrt(N))

    i = 1
    plt.figure(figsize=(32, 18))
    for type_, color in zip([-1, 0, 1], colors):
        for grid, params in pairs:
            T = params["T"]
            mu_v = params["mu_v"]

            ax = plt.subplot(side_dim, side_dim, i, projection='3d')
            plt.title(f"$T = {T}$; $\mu_v={mu_v}$", fontdict = {'fontsize' : 20})
            plot_point_cloud(grid, type_, ax, color, 1.0)

            for axis in [ax.axes.xaxis, ax.axes.yaxis, ax.axes.zaxis]:
                axis.set_ticklabels([])

                for line in axis.get_ticklines():
                    line.set_visible(False)

            i += 1

    plt.savefig("metro_grid_3d.pdf")


def main():
    set_style()

    graph = dg.Graph(dg.HDF5("data_cache"))

    graph.register_all(map(dg.task, [
        metro_grid_2D,
        metro_grid_3D,
        plot_metro_2D,
        plot_metro_3D
    ]))

    graph("plot_metro_3D")

if __name__ == '__main__':
    main()

