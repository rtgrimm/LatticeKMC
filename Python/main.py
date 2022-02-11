import Nano
from Python.wrap import VecToMat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def plot_point_cloud(image, key, ax):
    points = []

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for k in range(0, image.shape[2]):
                if(image[i, j, k] == key):
                    points.append((i, j, k))

    points = np.array(points)

    ax.set_proj_type("persp")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0)


def main():



    dim = Nano.IVec3D()
    dim.x = 50; dim.y = 50; dim.z = 50

    lattice = Nano.Lattice(dim)
    lattice.energy_map.set_uniform_binary(-1.0, 0.0)

    gen = Nano.RandomGenerator(123)
    lattice.uniform_init(gen)



    metropolis = Nano.Metropolis(lattice, gen)
    metropolis.step(int(1e6))

    output = VecToMat(lattice.get_types())
    grid = output.reshape((dim.x, dim.y, dim.z))


    #for i in range(0, 60, 10):
    #    plt.figure()
    #    plt.imshow(grid[:,:,i] - grid[:,:,0])
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_point_cloud(grid, 1, fig.gca())
    plot_point_cloud(grid, -1, fig.gca())
    plt.show()

    #run_kmc(dim, gen, lattice)


def run_kmc(dim, gen, lattice):
    sim_params = Nano.SimulationParams()
    sim_params.default_events()
    sim = Nano.Simulation(sim_params, lattice, gen)
    plt.ion()
    fig = plt.gcf()
    ax = fig.add_subplot(projection='3d')
    for i in range(0, 100):
        ax.clear()

        sim.step()

        output = VecToMat(lattice.get_types())
        grid = output.reshape((dim.x, dim.y, dim.z))
        # plt.imshow(grid[:,:,0])

        plot_point_cloud(grid, 1, ax)
        # plot_point_cloud(grid, -1, ax)

        print(f"{sim.get_time()}")
        print(f"A:{np.sum(grid == -1)}")
        print(f"A:{np.sum(grid == 1)}")

        plt.draw()
        plt.pause(1e-8)


if __name__ == '__main__':
    main()