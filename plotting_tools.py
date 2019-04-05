import numpy as np
import math # ceil
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import pylab


def make_plot(bodies, outfile=None, no_central=False):
    """Create a 3D plot of the system."""

    fig = plt.figure()
    # colours = ['r', 'b', 'g', 'y', 'm', 'c']

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # max_range = 0
    # zabs = 0
    for index, current_body in enumerate(bodies):
        if no_central is True and index == 0:
            ax.set_title(str(len(bodies_list)) + " Bodies (w/o Central Body)")
            continue
        ax.plot(current_body[:, 0], current_body[:, 1], current_body[:, 2],
                label=bodies_list[index].name)  # c=random.choice(colours), # current_body["name"]
        if no_central is False and index == len(bodies) - 1:
            ax.set_title(str(len(bodies_list)) + " Body System")
        # max_dim = max(max(abs(current_body[:, 0])), max(abs(current_body[:, 1])), max(abs(current_body[:, 2]))) # ['x']
        # if max_dim > max_range:
        #     max_range = max_dim
        # zabs_dim = max(abs(current_body[:, 2]))
        # if zabs_dim > zabs:
        #     zabs = zabs_dim
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0)) # Try both for z if others are bad
    ax.set_xlabel('X, m', rotation=-15) # fontsize=10
    ax.set_ylabel("Y, m", rotation=45)
    ax.set_zlabel("Z, m", rotation='vertical')
    # ax.set_xlim(-max_range, max_range)
    # ax.set_ylim(-max_range, max_range)
    # ax.set_zlim(-zabs, zabs) # ax.set_zlim(-max_range, max_range)
    ax.legend()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def make_plot_with_energy3d(bodies, potential, kinetic, name="Full System", outfile=None, no_central=False): # trajectories, trajectories[1][:, 0] = trajectories[1]['x']
    """3D plot with energy."""

    fig = plt.figure(figsize=plt.figaspect(0.5))
    # colours = ['r', 'b', 'g', 'y', 'm', 'c']

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # max_range = 0
    # zabs = 0
    for index, current_body in enumerate(bodies):
        if no_central is True and index == 0:
            ax.set_title(str(len(bodies_list)) + " Bodies (w/o Central Body)")
            continue
        ax.plot(current_body[:, 0], current_body[:, 1], current_body[:, 2],
                label=bodies_list[index].name)  # c=random.choice(colours),  current_body["name"]
        if no_central is False and index == len(bodies)-1:
            ax.set_title(str(len(bodies_list)) + " Body System")
        # max_dim = max(max(abs(current_body[:, 0])), max(abs(current_body[:, 1])), max(abs(current_body[:, 2]))) # ['x']
        # if max_dim > max_range:
        #     max_range = max_dim
        # zabs_dim = max(abs(current_body[:, 2]))
        # if zabs_dim > zabs:
        #     zabs = zabs_dim
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0)) # z
    ax.set_xlabel('X, m', rotation=-15) # fontsize=10
    ax.set_ylabel("Y, m", rotation=45)
    ax.set_zlabel("Z, m", rotation='vertical')
    # ax.set_xlim(-max_range, max_range)
    # ax.set_ylim(-max_range, max_range)
    # ax.set_zlim(-zabs, zabs)
    # ax.set_zlim(-max_range, max_range)
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(kinetic, label="Kinetic Energy")
    ax.plot(potential, label="Potential Energy")
    ax.plot((kinetic + potential), label="Total Energy")
    ax.set_title(name+" Energies")
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel('Steps')
    ax.set_ylabel("Energies, J")
    ax.legend()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

# make_plot_with_energy3d(trajectories, E_pot_tot, E_kin_tot) # , no_central=True)


def make_plot_with_energy2d(bodies, potential, kinetic, axis=(0, 1), name="Full System",
                            outfile=None, no_central=False): # trajectories, trajectories[1][:, 0] =trajectories[1]['x']
    """2D plot with energy."""

    fig = plt.figure(figsize=plt.figaspect(0.5))
    # colours = ['r', 'b', 'g', 'y', 'm', 'c']

    ax = fig.add_subplot(1, 2, 1)
    title2d = [("X", "Y", "Z")[i] for i in axis][0] + [("X", "Y", "Z")[i] for i in axis][1]
    # max_range = 0
    # zabs = 0

    for index, current_body in enumerate(bodies):
        if no_central is True and index == 0:
            ax.set_title(title2d + " Plot of " + str(len(bodies_list)) + " Bodies (w/o Central Body)")
            continue
        ax.plot(current_body[:, axis[0]], current_body[:, axis[1]], label=bodies_list[index].name)
        if no_central is False and index == len(bodies)-1:
            ax.set_title(title2d + " Plot of " + str(len(bodies_list)) + " Bodies")
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel(title2d[0] + ', m')
    ax.set_ylabel(title2d[1] + ', m')
    # ax.set_xlim(-max_range, max_range) # Not necessarily needed
    # if axis[1] == 2:
    #     ax.set_ylim(-zabs, zabs)
    # else:
    #     ax.set_ylim(-max_range, max_range)
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(kinetic, label="Kinetic Energy")
    ax.plot(potential, label="Potential Energy")
    ax.plot((kinetic + potential), label="Total Energy")
    ax.set_title(name + " Energies")
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel('Steps') # , fontsize=10
    ax.set_ylabel("Energies, J")
    ax.legend()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

# make_plot_with_energy2d(trajectories, bodies_list[0].E_potential, bodies_list[0].E_kinetic, bodies_list[0].name)
# axis_list = [(0, 1), (0, 2), (1, 2)]
# for i in axis_list:
#     make_plot_with_energy2d(trajectories, E_pot_tot, E_kin_tot, axis=i) # , no_central=True)
#     if index == len(bodies_list) -1:
#         E_pot_tot =
# for index, body in enumerate(bodies_list):
#     E_pot_tot += body.E_potential
#     E_kin_tot += body.E_kinetic
#     E_tot += body.E_total
# E_pot_tot = [x for x in bodies_list[0].E_kinetic + bodies_list[1].E_kinetic]


def make_super_plot(bodies, potential, kinetic, bodies_list, name="System", outfile=None, dontPlot=False, plotAllEs=False): # time,
    """Energy, 3D plot and each axis 2D plots."""

    # TODO: move legend and add title (& for XYZ plots?), change energy steps to time

    # colours = ['r', 'b', 'g', 'y', 'm', 'c']
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))  # maybe 8, 10# ax = fig.add_subplot(3,2,1)

    # Energy plot
    if plotAllEs is True:
        for index, current_body in enumerate(bodies_list):
            if dontPlot is not False and index in dontPlot:
                continue
            ax[0, 0].plot(current_body.E_kinetic, label=(current_body.name + "kinetic Energy"))
            ax[0, 0].plot(current_body.E_potential, label=(current_body.name + "potential Energy"))

    ax[0, 0].plot(kinetic, label="Kinetic Energy")
    ax[0, 0].plot(potential, label="Potential Energy")
    ax[0, 0].plot((kinetic + potential), label="Total Energy")
    ax[0, 0].set_title(name + " Energies")
    # ax[0, 0].set_xticks(time)
    ax[0, 0].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax[0, 0].set_xlabel('Steps')
    ax[0, 0].set_ylabel("Energies, J")

    Eh, El = ax[0, 0].get_legend_handles_labels()

    # 2D plots
    for i, subloc in enumerate([(1, 0), (2, 0), (2, 1)]): # 2D position plots
        xyzcoords = [(0, 1), (0, 2), (1, 2)][i]
        title2d = [("X", "Y", "Z")[j] for j in xyzcoords][0] + [("X", "Y", "Z")[j] for j in xyzcoords][1]
        # if i == 1:
        #     ax[subloc[0], subloc[1]].yaxis.tick_right()

        for index, current_body in enumerate(bodies):
            if dontPlot is not False and index in dontPlot:
                continue
            ax[subloc[0], subloc[1]].plot(current_body[:, xyzcoords[0]], current_body[:, xyzcoords[1]],
                                          label=bodies_list[index].name)
        ax[subloc[0], subloc[1]].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax[subloc[0], subloc[1]].set_xlabel(title2d[0] + ', m')
        ax[subloc[0], subloc[1]].set_ylabel(title2d[1] + ', m')


    # 3D plot
    ax[1, 1].remove() # Clear space for 3D plot
    # Corrects 3D plot positioning and size
    gs2 = gridspec.GridSpec(3, 2)
    gs2.update(left=-0.02, right=0.92, hspace=0.1)
    ax = plt.subplot(gs2[1, 1], projection='3d')

    # max_range = 0
    # zabs = 0
    # u = np.linspace(0, 2 * np.pi, 100) # For sphere plots
    # v = np.linspace(0, np.pi, 100)
    for index, current_body in enumerate(bodies):
        if dontPlot is not False and index in dontPlot:
            if index == dontPlot[-1]:
                ax.set_title(str(len(bodies_list)-len(dontPlot)) + " Bodies (" + str(len(dontPlot)) + " excluded)")
            continue

        ax.plot(current_body[:, 0], current_body[:, 1], current_body[:, 2],
                label=bodies_list[index].name)  # c=random.choice(colours),  current_body["name"]

        # x = current_body[-1, 0] + (bodies_list[index].radius * np.outer(np.cos(u), np.sin(v)))
        # y = current_body[-1, 1] + (bodies_list[index].radius * np.outer(np.sin(u), np.sin(v)))
        # z = current_body[-1, 2] + (bodies_list[index].radius * np.outer(np.ones(np.size(u)), np.cos(v)))
        # ax.plot_surface(x, y, z, linewidth=0) # Too small to notice for solar system

        if dontPlot is False and index == len(bodies) - 1:
            ax.set_title(str(len(bodies_list)) + " Body System")
        # max_dim = max(max(abs(current_body[:, 0])), max(abs(current_body[:, 1])), max(abs(current_body[:, 2]))) # ['x']
        # if max_dim > max_range:
        #     max_range = max_dim
        # zabs_dim = max(abs(current_body[:, 2]))
        # if zabs_dim > zabs:
        #     zabs = zabs_dim
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))  # z
    ax.set_xlabel('X, m', rotation=-15)  # fontsize=10
    ax.set_ylabel("Y, m", rotation=45)
    ax.set_zlabel("Z, m", rotation='vertical')
    # ax.set_xlim(-max_range, max_range)
    # ax.set_ylim(-max_range, max_range)
    # ax.set_zlim(-zabs, zabs)
    # ax.set_zlim(-max_range, max_range)
    # ax.legend()

    Bh, Bl = ax.get_legend_handles_labels()
    gs1 = gridspec.GridSpec(3, 2)
    BLplt = plt.subplot(gs1[0, 1])
    BLplt.axis('off')
    frstLgnd = BLplt.legend(Eh, El, loc='upper left', bbox_to_anchor=(-0.20, 1.0), fancybox=True, shadow=True)
    BLplt.add_artist(frstLgnd)
    BLplt.legend(Bh, Bl, loc='lower center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

    fig.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
