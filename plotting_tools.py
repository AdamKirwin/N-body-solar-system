import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import pylab


def make_plot(bodies, bodies_list, outfile=None, no_central=False):
    """Create a 3D plot of the system."""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for index, current_body in enumerate(bodies):
        if no_central is True and index == 0:
            ax.set_title(str(len(bodies_list)) + " Bodies (w/o Central Body)")
            continue

        ax.plot(current_body[:, 0], current_body[:, 1], current_body[:, 2], label=bodies_list[index].name)

        if no_central is False and index == len(bodies) - 1:
            ax.set_title(str(len(bodies_list)) + " Body System")

    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel('X, m', rotation=-15)
    ax.set_ylabel("Y, m", rotation=45)
    ax.set_zlabel("Z, m", rotation='vertical')
    ax.legend()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def make_plot_with_energy3d(bodies, potential, kinetic, bodies_list, name="Full System", outfile=None,
                            no_central=False):
    """3D plot and energy plot as subplots."""

    fig = plt.figure(figsize=plt.figaspect(0.5))
    # 3D subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for index, current_body in enumerate(bodies):
        if no_central is True and index == 0:
            ax.set_title(str(len(bodies_list)) + " Bodies (w/o Central Body)")
            continue
        ax.plot(current_body[:, 0], current_body[:, 1], current_body[:, 2], label=bodies_list[index].name)
        if no_central is False and index == len(bodies)-1:
            ax.set_title(str(len(bodies_list)) + " Body System")
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel('X, m', rotation=-15)
    ax.set_ylabel("Y, m", rotation=45)
    ax.set_zlabel("Z, m", rotation='vertical')
    ax.legend()

    # energy subplot
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(kinetic, label="Kinetic Energy")
    ax.plot(potential, label="Potential Energy")
    ax.plot((kinetic + potential), label="Total Energy")
    ax.set_title(name+" Energies")
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel('Steps')
    ax.set_ylabel("Energy, J")
    ax.legend()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def make_plot_with_energy2d(bodies, potential, kinetic, bodies_list, axis=(0, 1), name="Full System",
                            outfile=None, no_central=False):
    """2D plot and energy plot as subplots."""

    fig = plt.figure(figsize=plt.figaspect(0.5))
    #2D subplot
    ax = fig.add_subplot(1, 2, 1)
    title2d = [("X", "Y", "Z")[i] for i in axis][0] + [("X", "Y", "Z")[i] for i in axis][1]
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
    ax.legend()

    # energy subplot
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(kinetic, label="Kinetic Energy")
    ax.plot(potential, label="Potential Energy")
    ax.plot((kinetic + potential), label="Total Energy")
    ax.set_title(name + " Energies")
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel('Steps')
    ax.set_ylabel("Energy, J")
    ax.legend()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def make_super_plot(bodies, potential, kinetic, bodies_list, name="System", outfile=None, dontPlot=False,
                    plotAllEs=False):
    """Energy plot, 3D plot and XY, XZ, YZ 2D plots as subplots."""

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
    # energy subplot
    # TODO: change energy steps to time
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
    ax[0, 0].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax[0, 0].set_xlabel('Steps')
    ax[0, 0].set_ylabel("Energy, J")
    Eh, El = ax[0, 0].get_legend_handles_labels()  # gets energy legends for placement in different gridspace

    # 2D subplots
    for i, subloc in enumerate([(1, 0), (2, 0), (2, 1)]): # 2D position plots
        xyzcoords = [(0, 1), (0, 2), (1, 2)][i]
        title2d = [("X", "Y", "Z")[j] for j in xyzcoords][0] + [("X", "Y", "Z")[j] for j in xyzcoords][1]
        for index, current_body in enumerate(bodies):
            if dontPlot is not False and index in dontPlot:
                continue
            ax[subloc[0], subloc[1]].plot(current_body[:, xyzcoords[0]], current_body[:, xyzcoords[1]],
                                          label=bodies_list[index].name)
        ax[subloc[0], subloc[1]].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax[subloc[0], subloc[1]].set_xlabel(title2d[0] + ', m')
        ax[subloc[0], subloc[1]].set_ylabel(title2d[1] + ', m')

    # 3D subplot
    ax[1, 1].remove()  # Clear regular matplotlib gridspace for 3D plot using gridspec
    # custom places 3D plot to desired location using gridspec
    gs2 = gridspec.GridSpec(3, 2)
    gs2.update(left=-0.02, right=0.92, hspace=0.1)
    ax = plt.subplot(gs2[1, 1], projection='3d')
    for index, current_body in enumerate(bodies):
        if dontPlot is not False and index in dontPlot:
            if index == dontPlot[-1]:
                ax.set_title(str(len(bodies_list)-len(dontPlot)) + " Bodies (" + str(len(dontPlot)) + " excluded)")
            continue
        ax.plot(current_body[:, 0], current_body[:, 1], current_body[:, 2], label=bodies_list[index].name)
        if dontPlot is False and index == len(bodies) - 1:
            ax.set_title(str(len(bodies_list)) + " Body System")
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.set_xlabel('X, m', rotation=-15)
    ax.set_ylabel("Y, m", rotation=45)
    ax.set_zlabel("Z, m", rotation='vertical')
    Bh, Bl = ax.get_legend_handles_labels()  # gets bodies plot legend for placement in different gridspace

    # placement of legends outside of plots, within their own gridspace
    # TODO: add title to this space
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
