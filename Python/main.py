import Nano
from Python.Nano import vector_data
from Python.style import set_style, pal
from Python.wrap import int_vec_to_mat, double_vec_to_mat, IVec3D_to_mat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def plot_point_cloud(image, key, ax, c, s = 10.0):
    points = []

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for k in range(0, image.shape[2]):
                if(image[i, j, k] == key):
                    points.append((i, j, k))

    points = np.array(points)

    ax.set_proj_type("persp")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s, color=c)


def run_metro_test():
    dim = Nano.IVec3D()
    dim.x = 30; dim.y = 30; dim.z = 30

    lattice = Nano.Lattice(dim)
    lattice.energy_map.set_uniform_binary(-1.0, 0.0)

    gen = Nano.RandomGenerator(123)
    lattice.uniform_init(gen)

    metropolis = Nano.Metropolis(lattice, gen)
    metropolis.step(int(1e6))

    output = int_vec_to_mat(lattice.get_types())
    grid = output.reshape((dim.x, dim.y, dim.z))


    #for i in range(0, 60, 10):
    #    plt.figure()
    #    plt.imshow(grid[:,:,i] - grid[:,:,0])
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_point_cloud(grid, 1, fig.gca(), "red")

    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')
    plot_point_cloud(grid, -1, fig2.gca(), "blue")
    plt.show()


def set_model(lattice, mu_a, mu_b, mu_v, T, J_v_b = 0.0, J_v_a = 0.0, k_b = 1.0):
    lattice.energy_map.add_particle_type(-1)
    lattice.energy_map.add_particle_type(0)
    lattice.energy_map.add_particle_type(1)

    lattice.energy_map.set_interaction(1, 0, J_v_b)
    lattice.energy_map.set_interaction(-1, 0, J_v_a)
    lattice.energy_map.set_interaction(1, -1, 1.0)

    lattice.energy_map.set_interaction(1, 1, -1.0)
    lattice.energy_map.set_interaction(-1, -1, -1.0)
    lattice.energy_map.set_interaction(0, 0, 0.0)

    lattice.energy_map.set_field(1, mu_a)
    lattice.energy_map.set_field(-1, mu_b)
    lattice.energy_map.set_field(0, mu_v)

    lattice.energy_map.beta = 1 / (T * k_b)

def run_kmc_test():
    dim = Nano.IVec3D()
    dim.x = 1000; dim.y = 1000; dim.z = 1

    lattice = Nano.Lattice(dim)
    lattice.set_dim(Nano.LatticeDim_Two)

    set_model(lattice, 2.0, 2.0, 0.1625, 0.55, J_v_b=0.0, J_v_a=0.0)

    gen = Nano.RandomGenerator(123)
    lattice.uniform_init(gen)

    metropolis = Nano.Metropolis(lattice, gen)
    metropolis.step(int(1e7))

    lattice.energy_map.reset()
    set_model(lattice, 2.0, 2.0, 0.1625, 2.0, J_v_b=0.0, J_v_a=0.0)


    run_kmc(dim, gen, lattice)

def plot_particle_data(lattice, x, y, z):
    point = Nano.IVec3D()
    point.x = x
    point.y = y
    point.z = z

    particle = lattice.particle_at(point)

    if(particle.hop_list.size() == 0):
        return

    hops = IVec3D_to_mat(particle.hop_list)
    hops = np.reshape(hops, (len(hops) // 3, 3))
    hops = np.cumsum(hops, axis=0)

    plt.plot(x + hops[:, 0], y + hops[:, 1])


def plot_MSD(lattice, sim, type, color, name):
    msd_estimate = Nano.MSDEstimate(lattice)

    msd_estimate.run(sim.get_time(), 1e-2, type)
    series = msd_estimate.get_MSD_series()

    times = double_vec_to_mat(series.times)
    means = double_vec_to_mat(series.values)

    plt.scatter(times, means, s=10.0, color=color)
    plt.plot(times, means, color=color, label=name)

def run_kmc(dim, gen, lattice):
    sim_params = Nano.SimulationParams()
    sim_params.default_events()
    sim = Nano.Simulation(sim_params, lattice, gen)

    plt.ion()
    fig = plt.gcf()
    #ax = fig.add_subplot(projection='3d')

    lattice.enable_particle_tracking()

    for i in range(0, 1000):
        #ax.clear()
        plt.clf()

        sim.multi_step(10000)

        colors = pal("bright", 3)

        plot_MSD(lattice, sim, 1, next(colors), "A")
        plot_MSD(lattice, sim, 0, next(colors), "V")
        plot_MSD(lattice, sim, -1, next(colors), "B")
        plt.legend(frameon=False)
        plt.ylabel("MSD")
        plt.xlabel("Time")

        #for i in range(0, 50):
        #    for j in range(0, 50):
        #        plot_particle_data(lattice, i, j, 0)

        #output = int_vec_to_mat(lattice.get_types())
        #grid = output.reshape((dim.x, dim.y, dim.z))
        #plt.imshow(grid[:,:,0])

        #plot_point_cloud(grid, -1, ax, "green", s=1.0)
        #plot_point_cloud(grid, 1, ax, "blue", s=1.0)
        #plot_point_cloud(grid, 0, ax, "red", s=1.0)


        plt.draw()
        plt.pause(1e-8)



if __name__ == '__main__':
    set_style()
    run_kmc_test()