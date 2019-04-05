import numpy as np
import math
import mpmath as mp

mp.mp.dps = 30
G_const = 6.67408e-11 # Gravitational Constant [m^3 kg^-1 s^-2]


def calculate_energy(body_index, trajectories, bodies_list): # Incorporate into compute_simulation?
    """Calculates the energies of a body from all bodies acting on it."""

    focus_body = trajectories[body_index]  # self.bodies
    potential = 0
    kinetic = 0

    for index, other_body in enumerate(trajectories):  # self.bodies
        if index != body_index:
            dx = (other_body[:, 0] - focus_body[:, 0])  # ['x']
            dy = (other_body[:, 1] - focus_body[:, 1])
            dz = (other_body[:, 2] - focus_body[:, 2])
            r_between = ((dx * dx) + (dy * dy) + (dz * dz))
            r_between = np.sqrt(r_between)

            pot_tmp = G_const * bodies_list[index].mass / r_between  # Quicker calculation
            potential += -pot_tmp * bodies_list[body_index].mass  # focus_body.mass

            dvx = (other_body[:, 3] - focus_body[:, 3])
            dvy = (other_body[:, 4] - focus_body[:, 4])
            dvz = (other_body[:, 5] - focus_body[:, 5])
            v_between = ((dvx * dvx) + (dvy * dvy) + (dvz * dvz))
            kinetic += (bodies_list[body_index].mass / 2) * v_between

    return potential, kinetic


# https://stackoverflow.com/questions/34560620/how-do-i-plot-a-planets-orbit-as-a-function-of-time-on-an-already-plotted-ellip
# Equations taken from:
# https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time
def kepler_calc(rmax, rmin, Mbody, t):
    EPSILON = 1e-15  # calculation precision, # was 1e-12

    def solve_bisection(fn, xmin, xmax, epsilon=EPSILON):
        while True:
            xmid = (xmin + xmax) * 0.5
            if (xmax - xmin < epsilon):
                return xmid
            fn_mid = fn(xmid)
            fn_min = fn(xmin)
            if fn_min * fn_mid < 0:
                xmax = xmid
            else:
                xmin = xmid

    mu = G_const * Mbody # standard gravitational parameter
    eps = (rmax - rmin) / (rmax + rmin) # eccentricity
    p = rmin * (1 + eps) # semi-latus rectum
    a = p / (1 - eps ** 2) # semi/half major axis
    P = math.sqrt(a ** 3 / mu) # period
    if t / P == math.pi:  # fix to stop midpoint returning start values (x isnt perfect)
        t = t * (2.000000000000001 / 2)
    M = (t / P) % (2 * math.pi) # mean anomaly

    def fn_E(E): # eccentric anomaly
        return M - (E - eps * math.sin(E))
    E = solve_bisection(fn_E, 0, 2 * math.pi)

    def fn_theta(theta): # true anomaly, TODO: what if E == pi?
        return (1 - eps) * math.tan(theta / 2) ** 2 - ((1 + eps) * math.tan(E / 2) ** 2)
    theta = solve_bisection(fn_theta, 0, math.pi)

    if (E > math.pi): # if we are at the second half of the orbit
        theta = 2 * math.pi - theta
    r = a * (1 - eps * math.cos(E)) # heliocentric distance

    x = -r * math.sin(-theta)
    y = r * math.cos(theta)

    return x, y


def kepler_calc2(r, t, T):
    """Shorter version of above code, assumes circular orbit."""

    # TODO: fix for t/P==pi?
    n = (2 * math.pi) / T
    theta = n * t
    x = -r * math.sin(-theta)
    y = r * math.cos(theta)

    return x, y

# pos_x1, pos_y1 = kepler_calc2(bodies_list[1].position.y, time_step*steps)

def kepler_calc3(r, t, T):
    """precise version of the above shorter code, assuming circular orbit."""

    mp.mp.dps = 30 #~16+ good enough
    # TODO: fix for t/P==pi?
    n = (2 * mp.pi) / T
    theta = n * t
    x = -r * mp.sin(-theta)
    y = r * mp.cos(theta)

    return x, y
