import numpy as np
import math
import mpmath as mp

mp.mp.dps = 30
G_const = 6.67408e-11 # Gravitational Constant [m^3 kg^-1 s^-2]


def compute_simulation(integrator, number_of_steps, report_freq, bodies_list):
    """The main function for performing the necessary steps of simulation."""

    n_append = int(number_of_steps / report_freq) + 1  # scaled factor for appending new trajectories to body data

    # creation of trajectory 'history' array
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
        # simulation performed through integration steps, data stored and completion % printed in integer steps
        integrator.perform_integration()
        if i % report_freq == 0:
            append_i += 1
            for index, body_position in enumerate(body_positional_hist):
                body_position[append_i] = [bodies_list[index].position.x, bodies_list[index].position.y,
                                           bodies_list[index].position.z, bodies_list[index].velocity.x,
                                           bodies_list[index].velocity.y, bodies_list[index].velocity.z]

            compScale += compScale0
            if int(compScale) > int(compScale - compScale0):
                print("Computation is " + str(int(compScale)) + "% complete")

    return body_positional_hist


def calculate_energy(body_index, trajectories, bodies_list):
    """Calculates the energies of a body for all reported trajectories from a simulation."""

    focus_body = trajectories[body_index]
    potential = 0
    kinetic = 0
    for index, other_body in enumerate(trajectories):
        # calculation of a bodies energy over simulation using reported trajectory history of all bodies
        if index != body_index:
            dx = (other_body[:, 0] - focus_body[:, 0])
            dy = (other_body[:, 1] - focus_body[:, 1])
            dz = (other_body[:, 2] - focus_body[:, 2])
            r_between = ((dx * dx) + (dy * dy) + (dz * dz))
            r_between = np.sqrt(r_between)

            pot_tmp = G_const * bodies_list[index].mass / r_between
            potential += -pot_tmp * bodies_list[body_index].mass

            dvx = (other_body[:, 3] - focus_body[:, 3])
            dvy = (other_body[:, 4] - focus_body[:, 4])
            dvz = (other_body[:, 5] - focus_body[:, 5])
            v_between = ((dvx * dvx) + (dvy * dvy) + (dvz * dvz))
            kinetic += (bodies_list[body_index].mass / 2) * v_between

    return potential, kinetic
