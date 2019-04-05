import numpy as np
import math
import mpmath as mp
# import matplotlib.pyplot as plt
# import random
# from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib.gridspec import GridSpec
# from matplotlib import gridspec
# import pylab
import time
import sys

sys.path.append('C:/Users/Adam/PycharmProjects/FinalProject/Scripts')
from integrators import *
from helpers import *
from plotting_tools import *

mp.mp.dps = 30  # overkill to ensure precision, ~16-20+ actually necessary
G_const = 6.67408e-11  # Gravitational Constant [m^3 kg^-1 s^-2]


def compute_simulation(integrator, number_of_steps, report_freq):
    n_append = int(number_of_steps / report_freq) + 1

    body_positional_hist = []
    # TODO: use spheres of influence for determining how many calculations each body requires
    for index, current_body in enumerate(bodies_list):
        body_positional_hist.append(np.zeros((n_append + 1, 6)))
        body_positional_hist[index][0] = [current_body.position.x, current_body.position.y, current_body.position.z,
                                          current_body.velocity.x, current_body.velocity.y, current_body.velocity.z]

    append_i = 0
    compScale = 0
    compScale0 = 100. / n_append
    for i in range(0, number_of_steps + 1):
        integrator.perform_integration()
        if i % report_freq == 0:
            append_i += 1
            for index, body_position in enumerate(body_positional_hist):
                body_position[append_i] = [bodies_list[index].position.x, bodies_list[index].position.y,
                                           bodies_list[index].position.z, bodies_list[index].velocity.x,
                                           bodies_list[index].velocity.y, bodies_list[index].velocity.z]
                # need to write positions to file for live plot?

            compScale += compScale0
            if int(compScale) > int(compScale - compScale0):
                print("Computation is " + str(int(compScale)) + " % complete")

    return body_positional_hist


if __name__ == "__main__":

    # time_step_list = [0.0001, 0.0002, 0.0004, 0.0007,
    #                   0.001, 0.002, 0.004, 0.007,
    #                   0.01, 0.02, 0.04, 0.07,
    #                   0.1, 0.2, 0.4, 0.7,
    #                   1.0, 2.0, 4.0, 7.0,
    #                   10.0, 20.0, 40.0, 70.0,
    #                   100.0] # for KSP analysis
    time_step_list = [1000000.] # Set as largest variable timestep

    # integrators_list = [EulerMethod, EulerCromerMethod, EulerRichardsonMethod, RK4Method, VelocityVerletMethod,
    #                     VariableLeapfrogMethodM]  # TODO: StormerVerletMethod?
    integrators_list = [VariableLeapfrogMethod]

    for integrator_current in integrators_list:
        print(integrator_current)
        dist_err = []
        dist_err_prcnt = []
        dist_y_err = []
        dist_y_err_prcnt = []
        time_lst = []

        for index, step in enumerate(time_step_list):
            # TODO: create true N-body simulation with random bodies
            # solar_system = {
            #     'earth': {"mass": 5.972365356723323e+24, "position": VectorData((0, 0, 0)),
            #               "velocity": VectorData((0, 0, 0))},
            #     'satellite': {"mass": 0, "position": VectorData((0, 6734434.594057525, 0)),
            #                   "velocity": VectorData((7693.400089735311, 0, 0))},
            #     'kerbin': {"mass": 5.291515834392155e+22, "position": VectorData((0, 0, 0)),
            #                "velocity": VectorData((0, 0, 0))},
            #     'ksat': {"mass": 0, "position": VectorData((0, 675000, 0)),
            #              "velocity": VectorData((2287.3565528793274, 0, 0))},
            #     'earthC': {"mass": 5.972365356723323e+24, "position": VectorData((0, 149598023000, 0)),
            #                "velocity": VectorData((29780, 0, 0))},
            #     'satelliteC': {"mass": 0, "position": VectorData((0, 1e10, 0)),
            #                    "velocity": VectorData((3e4, 0, 0))}
            # } # for KSP analysis
            # Barycentric JPL DE421 data
            # TODO: base timestep on body type and orbital radius
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
                            "time_step": 100000.},
                'venus': {"mass": 4.8675e24,
                          "position": VectorData((1.09128163e+10, -9.76450007e+10, -4.46419427e+10)),
                          "velocity": VectorData((34606.55659866, 3813.19288143, -474.42681908)),
                          "radius": 6051800.,
                          "time_step": 100000.},
                'earth': {"mass": 5.972365356723323e24,
                          "position": VectorData((-2.59449269e+10, 1.33651814e+11, 5.79193563e+10)),
                          "velocity": VectorData((-29793.27054748, -4965.02845987, -2152.81505951)),
                          "radius": 6371000.,
                          "time_step": 100000.},
                'moon': {"mass": 1.,
                          "position": VectorData((-2.59110756e+10, 1.33987401e+11, 5.80375519e+10)),
                          "velocity": VectorData((-30888.88497398, -4910.13099286, -2058.54759181)),
                         "radius": 1737100.,
                         "time_step": 10000.},
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

            # bodies_list = ['kerbin', 'ksat'] # for KSP analysis
            bodies_list = ['sun', 'mercury', 'venus', 'earth', 'moon', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
            # , 'pluto']
            bodies_list[:] = [Body(mass=solar_system[i]["mass"], position=solar_system[i]["position"],
                                   velocity=solar_system[i]["velocity"], time_step=solar_system[i]["time_step"],
                                   name=i) for i in bodies_list]
            # radius=solar_system[i]["radius"], time_step=step,

            startTime = time.time()

            time_step = step
            # TODO: if integrator_current == VariableLeapfrogMethod:
            new_step = 0.  # For variable timestep methods
            for bodies in bodies_list:
                if bodies.time_step <= step and bodies.time_step > new_step:
                    new_step = bodies.time_step
            time_step = new_step

            T_orbit = 5206463427.  # [s] # N = 5206463427. # E= 32000000.
            # for KSP analysis
            # T_orbit = 2 * mp.pi * mp.sqrt((bodies_list[1].position.y * bodies_list[1].position.y
            #                                * bodies_list[1].position.y) / (G_const * bodies_list[0].mass))
            # TODO: keep steps above 100
            # steps = int(math.ceil(float(T_orbit) / time_step))  # ceil covers whole orbit # for ksp analysis
            steps = int(math.ceil(float(T_orbit) / new_step))  # ceil covers whole orbit
            Xrf = 5.2  # 320.
            rf = int(steps / Xrf)  # rf never less than 1, ?never more than steps/10?
            while (steps / 10.) < rf:  # ensures enough updates
                rf /= 10
            if int(rf) == 0:
                rf = 1
            # rf = 1  # updates trajectories list at every step # for KSP analysis
            print("Report frequency is " + str(rf))

            # Kepler orbit calculation of expected final location
            # pos_x1, pos_y1 = kepler_calc(bodies_list[1].position.y, bodies_list[1].position.y,
            #                                         bodies_list[0].mass, time_step*steps)
            # pos_x1, pos_y1 = kepler_calc3(bodies_list[1].position.y, time_step * steps, T_orbit) # For KSP analysis

            integrator = integrator_current(time_step=time_step, bodies_list=bodies_list)
            trajectories = compute_simulation(integrator, number_of_steps=steps, report_freq=rf)

            endTime = time.time() - startTime
            time_lst.append(endTime)

            E_pot_tot = 0
            E_kin_tot = 0
            E_tot = 0
            for index, body in enumerate(bodies_list):
                body.E_potential, body.E_kinetic = calculate_energy(index, trajectories, bodies_list)
                body.E_total = body.E_kinetic + body.E_potential
                E_pot_tot += body.E_potential
                E_kin_tot += body.E_kinetic
                E_tot += body.E_total

            # timetmp = len(E_pot_tot)
            # timetmp2 = T_orbit / len(E_pot_tot)
            # time = [(timetmp2*(i+1)) for i in range(timetmp)]
            make_super_plot(trajectories, E_pot_tot, E_kin_tot, bodies_list)

            # integratorname = integrator.__class__.__name__[:-6]  # Provided the integrator ends in 'Method'
            # Compare distance errors for given timestep # for KSP analysis
            # TODO: Sort to Func?
        #     pos_z1 = 0.
        #     # for index, val in enumerate(trajectories[1][:, 0][::-1]): # trajectories[1]['x'][::-1]
        #     #     # for finding smallest of final values
        #     #     if index == 0:
        #     #         continue
        #     #     elif abs(val) > trajectories[1][:, 0][::-1][index - 1]:
        #     #         pos_x2 = trajectories[1][:, 0][::-1][index - 1]
        #     #         pos_y2 = trajectories[1][:, 1][::-1][index - 1]
        #     #         pos_z2 = trajectories[1][:, 2][::-1][index - 1]
        #     #         break
        #     pos_x2 = trajectories[1][:, 0][-1]
        #     pos_y2 = trajectories[1][:, 1][-1]
        #     pos_z2 = trajectories[1][:, 2][-1]
        #     dx = pos_x2 - pos_x1
        #     dy = pos_y2 - pos_y1
        #     dz = pos_z2 - pos_z1
        #     dist = math.sqrt((dx * dx) + (dy * dy))  # + (dz * dz)
        #     dist_prcnt = dist / pos_y1
        #     dist_err.append(dist)
        #     dist_err_prcnt.append(dist_prcnt)
        #     dist_y = math.sqrt((dy * dy))  # (dx * dx) +
        #     dist_y_prcnt = dist / pos_y1
        #     dist_y_err.append(dist_y)
        #     dist_y_err_prcnt.append(dist_y_prcnt)
        #     print(integratorname, time_step, dist, dist_prcnt, dist_y, dist_y_prcnt, endTime)
        #     # if dist_prcnt >= 0.1:
        #     #     break
        #
        # with open((integratorname + "_err.txt"), "w") as f:
        #     for (dt, err, err_prcnt, erry, erry_prcnt, timelst) in zip(time_step_list, dist_err, dist_err_prcnt,
        #                                                                dist_y_err, dist_y_err_prcnt, time_lst):
        #         f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(dt, err, err_prcnt, erry, erry_prcnt, timelst))

    # for KSP analysis
    # orbital_radius = [675000.,
    #                   1000000., 2000000., 4000000., 7000000.,
    #                   10000000.0, 20000000.0, 40000000.0, 70000000.0,
    #                   100000000.0]
    #
    # dist_err = []
    # dist_err_prcnt = []
    # dist_y_err = []
    # dist_y_err_prcnt = []
    # time_lst = []
    # ts_lst = []
    # orb_rad_lst = []
    #
    # for orbRad in orbital_radius:
    #     for index, step in enumerate(time_step_list):
    #         solar_system = {
    #             'earth': {"mass": 5.972365356723323e+24, "position": VectorData((0, 0, 0)),
    #                       "velocity": VectorData((0, 0, 0))},
    #             'satellite': {"mass": 0, "position": VectorData((0, 6734434.594057525, 0)),
    #                           "velocity": VectorData((7693.400089735311, 0, 0))},
    #             'kerbin': {"mass": 5.291515834392155e+22, "position": VectorData((0, 0, 0)),
    #                        "velocity": VectorData((0, 0, 0))},
    #             'ksat': {"mass": 0, "position": VectorData((0, 675000, 0)),
    #                      "velocity": VectorData((2287.3565528793274, 0, 0))},
    #             'earthC': {"mass": 5.972365356723323e+24, "position": VectorData((0, 149598023000, 0)),
    #                        "velocity": VectorData((29780, 0, 0))},
    #             'satelliteC': {"mass": 0, "position": VectorData((0, 1e10, 0)),
    #                            "velocity": VectorData((3e4, 0, 0))}
    #         }
    #
    #         GM_const = G_const * 5.291515834392155e+22  # Kerbin # bodies_list[0].masss
    #         v_orb = mp.sqrt(GM_const / orbRad)
    #
    #         startTime = time.time()
    #
    #         bodies_list = ['kerbin', 'ksat']  # ['sun', 'earth', 'mars', 'marssat2', 'jupiter']
    #         bodies_list[:] = [Body(mass=solar_system[i]["mass"], position=solar_system[i]["position"],
    #                                velocity=solar_system[i]["velocity"], time_step=step, name=i) for i in bodies_list]
    #
    #         time_step = step
    #         T_orbit = 2 * mp.pi * mp.sqrt((orbRad * orbRad * orbRad) / GM_const)
    #         steps = int(math.ceil(float(T_orbit) / time_step))
    #         rf = 1
    #
    #         pos_x1, pos_y1 = kepler_calc3(bodies_list[1].position.y, time_step * steps, T_orbit)
    #
    #         integrator = VelocityVerletMethodM(time_step=time_step, bodies_list=bodies_list)  # integrator_current
    #         trajectories = compute_simulation(integrator, number_of_steps=steps, report_freq=rf)
    #
    #         endTime = time.time() - startTime
    #         time_lst.append(endTime)
    #
    #         integratorname = integrator.__class__.__name__[:-6]  # Provided the integrator ends in 'Method'
    #
    #         pos_z1 = 0.
    #         # for index, val in enumerate(trajectories[1][:, 0][::-1]):  # trajectories[1]['x'][::-1]
    #         #     if index == 0:
    #         #         continue
    #         #     elif abs(val) > trajectories[1][:, 0][::-1][index - 1]:
    #         #         pos_x2 = trajectories[1][:, 0][::-1][index - 1]
    #         #         pos_y2 = trajectories[1][:, 1][::-1][index - 1]
    #         #         pos_z2 = trajectories[1][:, 2][::-1][index - 1]
    #         #         break
    #         pos_x2 = trajectories[1][:, 0][-1]
    #         pos_y2 = trajectories[1][:, 1][-1]
    #         pos_z2 = trajectories[1][:, 2][-1]
    #         dx = pos_x2 - pos_x1
    #         dy = pos_y2 - pos_y1
    #         dz = pos_z2 - pos_z1
    #         dist = math.sqrt((dx * dx) + (dy * dy))  # + (dz * dz)
    #         dist_prcnt = dist / pos_y1
    #         dist_err.append(dist)
    #         dist_err_prcnt.append(dist_prcnt)
    #         dist_y = math.sqrt((dy * dy))  # (dx * dx) +
    #         dist_y_prcnt = dist / pos_y1
    #         dist_y_err.append(dist_y)
    #         dist_y_err_prcnt.append(dist_y_prcnt)
    #         ts_lst.append(step)
    #         orb_rad_lst.append(orbRad)
    #
    #         print(orbRad, step, dist, dist_prcnt, dist_y, dist_y_prcnt, endTime)
    #         # if dist_prcnt >= 0.1:
    #         #     break
    #
    # with open((integratorname + "_rad_ts.txt"), "w") as f:
    #     for (orbRadius, dt, err, err_prcnt, erry, erry_prcnt, timelst) in zip(orb_rad_lst, ts_lst,
    #                                                                           dist_err, dist_err_prcnt, dist_y_err,
    #                                                                           dist_y_err_prcnt, time_lst):
    #         f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(orbRadius, dt, err, err_prcnt, erry, erry_prcnt,
    #                                                              timelst))
