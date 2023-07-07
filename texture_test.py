from lookup_table import *
import matplotlib.pyplot as plt
import numpy as np
from orbit_math import *


# table for E
# M range from 0 to 2pi
# e range from 0 to 1


# # table for H
# # M range from -50 to 50
# # e range from 1 to inf


def test():


    TEST_SIZE = 2000
    x_gpu = ti.field(ti.f32, shape=(TEST_SIZE))
    y_gpu = ti.field(ti.f32, shape=(TEST_SIZE))
    z_gpu = ti.field(ti.f32, shape=(TEST_SIZE, TEST_SIZE))

    @ti.kernel
    def lookup(solver: ti.template(),origin: ti.template()):
        for i in range(TEST_SIZE):
            for j in range(TEST_SIZE):
                z_gpu[i, j] = solver.lookup(x_gpu[i], y_gpu[j]) - origin(x_gpu[i], y_gpu[j])

    solverE = lookup_table_2d((0, 2*np.pi), (0, 1), (1024, 1024),
                              warp_mode=(warp_mode_repeat_periodic,warp_mode_clamp),
                              periodic_offset=(2*np.pi,0))
    solverE.init_table(kepler_solver_E_iter)
    solverH = lookup_table_2d((-50, 50), (1, 10), (1024, 1024))
    solverH.init_table(kepler_solver_H_iter)

    # direct show table

    # # show table for E

    # plt.imshow(solverE.table.to_numpy(), cmap='rainbow')
    # plt.xlabel('e')
    # plt.ylabel('M')
    # plt.title('E(M,e)')
    # plt.show()

    # # # show table for H

    # plt.imshow(solverH.table.to_numpy(), cmap='rainbow')
    # plt.xlabel('e')
    # plt.ylabel('M')
    # plt.title('H(M,e)')
    # plt.show()

    # show two table in 3D plot with color
    # make 2 subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # make data
    x = np.linspace(-4*np.pi, 4*np.pi, TEST_SIZE)
    y = np.linspace(0, 1, TEST_SIZE)
    X, Y = np.meshgrid(y, x)
    x_gpu = ti.field(ti.f32, shape=(TEST_SIZE))
    y_gpu = ti.field(ti.f32, shape=(TEST_SIZE))
    z_gpu = ti.field(ti.f32, shape=(TEST_SIZE, TEST_SIZE))
    x_gpu.from_numpy(x)
    y_gpu.from_numpy(y)
    lookup(solverE,kepler_solver_E_iter)
    Z = z_gpu.to_numpy()
    # plot the surface
    ax1.plot_surface(X, Y, Z, cmap='rainbow')
    ax1.set_xlabel('e')
    ax1.set_ylabel('M')
    ax1.set_zlabel('E')
    ax1.set_title('E(e,M)')
    # make data
    x = np.linspace(-55, 55, TEST_SIZE)
    y = np.linspace(1, 12, TEST_SIZE)
    X, Y = np.meshgrid(y, x)
    x_gpu.from_numpy(x)
    y_gpu.from_numpy(y)
    lookup(solverH,kepler_solver_H_iter)
    Z = z_gpu.to_numpy()
    # plot the surface
    ax2.plot_surface(X, Y, Z, cmap='rainbow')
    ax2.set_xlabel('e')
    ax2.set_ylabel('M')
    ax2.set_zlabel('H')
    ax2.set_title('H(e,M)')
    # show the plot
    plt.show()
