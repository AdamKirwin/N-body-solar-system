import numpy as np
import math
import mpmath as mp
from integrators import *
from helpers import *
from plotting_tools import *

mp.mp.dps = 30  # overkill to ensure precision, ~16-20+ actually necessary
G_const = 6.67408e-11  # Gravitational Constant [m^3 kg^-1 s^-2]


def run_simulation():
    """Primary function for computation of solar system simulation.
    Use of one or more of the numerical solutions to Newtonian and Hamiltonian mechanics.
    Simulation using JPL DE421 solar system data.
    Calculates trajectories for fixed or variable timesteps, records trajectories with some report frequency (rf),
    plots energy, 2D plots and 3D plot within one image."""

    # integrator(s) selected for simulation
    # integrators_list = [EulerMethod, EulerCromerMethod, EulerRichardsonMethod, RK4Method, VelocityVerletMethod,
    #                     VariableLeapfrogMethodM]  # TODO: Stormer-Verlet?, Barnes-Hut
    integrators_list = [VariableLeapfrogMethod]

    for integrator_current in integrators_list:
        print(integrator_current)
        # TODO: increase number of bodies (with random data) to test scaling
        # TODO: automate timestep from body type and orbital radius
        # TODO: program collision detection and actions using body radii

        # Barycentric JPL DE421 data, from 01/01/2018
        # ~stationary central bodies can use the largest timestep, other timesteps based on orbital radii
        # TODO: function for providing date input and returning DE421 data for use within simulation
        solar_system = {
            'sun': {"mass": 1.9884754156078441e30,
                    "position": VectorData((2.69681288e+08, 8.48575856e+08, 3.48301564e+08)),
                    "velocity": VectorData((-10.1266145, 7.88267273, 3.68509522)),
                    "radius": 695700000.,
                    "time_step": 1000000.},
            'mercury': {"mass": 3.3011e23,
                        "position": VectorData((-5.77312729e+10, -2.30147560e+09,  4.67814111e+09)),
                        "velocity": VectorData((-9163.92901026, -41575.01229291, -21260.60227473)),
                        "radius": 2439700.,
                        "time_step": 10000.},
            'venus': {"mass": 4.8675e24,
                      "position": VectorData((1.09128163e+10, -9.76450007e+10, -4.46419427e+10)),
                      "velocity": VectorData((34606.55659866, 3813.19288143, -474.42681908)),
                      "radius": 6051800.,
                      "time_step": 100000.},
            'earth': {"mass": 5.972365356723323e24,
                      "position": VectorData((-2.59449269e+10, 1.33651814e+11, 5.79193563e+10)),
                      "velocity": VectorData((-29793.27054748, -4965.02845987, -2152.81505951)),
                      "radius": 6371000.,
                      "time_step": 10000.},
            'moon': {"mass": 7.349132015199098e+22,
                      "position": VectorData((-2.59110756e+10, 1.33987401e+11, 5.80375519e+10)),
                      "velocity": VectorData((-30888.88497398, -4910.13099286, -2058.54759181)),
                     "radius": 1737100.,
                     "time_step": 1000.},
            'mars': {"mass": 6.417119662934817e23,
                     "position": VectorData((-2.36643922e+11, -5.43759075e+10, -1.85864717e+10)),
                     "velocity": VectorData((6674.9298644, -19437.2310635, -9095.7608759)),
                     "radius": 3389500.,
                    "time_step": 100000.},
            'jupiter': {"mass": 1.8982e27,
                        "position": VectorData((-6.37206785e+11, -4.67834825e+11, -1.85023484e+11)),
                        "velocity": VectorData((7940.09091301, -8785.82229198, -3959.09881557)),
                        "radius": 69911000.,
                    "time_step": 1000000.},
            'saturn': {"mass": 5.6834e26,
                       "position": VectorData((7.16584891e+09, -1.39065140e+12, -5.74720444e+11)),
                       "velocity": VectorData((9128.82502048, 158.29967198, -327.67320294)),
                       "radius": 58232000.,
                    "time_step": 1000000.},
            'uranus': {"mass": 8.6810e25,
                       "position": VectorData((2.65136512e+12, 1.25558877e+12, 5.12413928e+11)),
                       "velocity": VectorData((-3150.40876942, 5246.93718077, 2342.57130328)),
                       "radius": 25362000.,
                    "time_step": 1000000.},
            'neptune': {"mass": 1.02413e26,
                        "position": VectorData((4.29071172e+12, -1.15042931e+12, -5.77700394e+11)),
                        "velocity": VectorData((1523.52962745, 4863.56479607, 1952.75422039)),
                        "radius": 24622000.,
                    "time_step": 1000000.},
            'pluto': {"mass": 1.303e22,
                      "position": VectorData((1.61240475e+12, -4.36530936e+12, -1.84809257e+12)),
                      "velocity": VectorData((5262.12369632, 1193.19124627, -1213.11300584)),
                      "radius": 1188300.,
                      "time_step": 1000000.}
        }

        time_step = 1000000. # Set as largest variable timestep
        # list bodies to be included within simulation and create list of body classes
        bodies_list = ['sun', 'mercury', 'venus', 'earth', 'moon', 'mars',
                       'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
        bodies_list[:] = [Body(mass=solar_system[i]["mass"], position=solar_system[i]["position"],
                               velocity=solar_system[i]["velocity"], time_step=solar_system[i]["time_step"],
                               name=i) for i in bodies_list]
        # time_step=time_step , for non-variable timestep
        # TODO: if integrator_current == VariableLeapfrogMethod: (do variable setup)
        # simulation setup parameters
        new_step = 0.  # For variable timestep methods
        for bodies in bodies_list:
            if bodies.time_step <= time_step and bodies.time_step > new_step:
                new_step = bodies.time_step
        T_orbit = 5206463427. # [s] # M: 7600521.6, E: 31536000., M: 59354353.76506807, N: 5206463427.
        # TODO: Choose report frequency based on available system memory?
        steps = int(math.ceil(float(T_orbit) / new_step))
        Xrf = 52
        rf = int(steps / Xrf)
        while (steps / 10.) < rf:  # ensures enough updates
            rf /= 10
        if int(rf) == 0:
            rf = 1
        rf = int(rf)
        print("Report frequency is " + str(rf))

        # setup integration method class and compute simulation
        integrator = integrator_current(time_step=new_step, bodies_list=bodies_list) # time_step
        trajectories = compute_simulation(integrator, number_of_steps=steps, report_freq=rf,
                                          bodies_list=bodies_list)

        # calculate energy for plotting
        E_pot_tot = 0
        E_kin_tot = 0
        E_tot = 0
        for index, body in enumerate(bodies_list):
            body.E_potential, body.E_kinetic = calculate_energy(index, trajectories, bodies_list)
            body.E_total = body.E_kinetic + body.E_potential
            E_pot_tot += body.E_potential
            E_kin_tot += body.E_kinetic
            E_tot += body.E_total

        make_super_plot(trajectories, E_pot_tot, E_kin_tot, bodies_list)


if __name__ == "__main__":
    run_simulation()