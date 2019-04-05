import math
import numpy as np
import mpmath as mp

mp.mp.dps = 30
G_const = 6.67408e-11  # Gravitational Constant [m^3 kg^-1 s^-2]


class VectorData:
    """Class for 3D vector data."""

    def __init__(self, point=(0,0,0)):
        self.x, self.y, self.z = float(point[0]), float(point[1]), float(point[2])


class Body:
    """Class representing a typical object and properties."""

    def __init__(self, mass, position, velocity, old_position = None, old_velocity = None, radius=None, time_step=None,
                 E_potential = None, E_kinetic = None, E_total = None, name=""):

        self.mass = mass
        self.position = position
        self.velocity = velocity

        self.old_position = old_position
        self.old_velocity = old_velocity

        self.radius = radius
        self.time_step = time_step
        self.body_time = 0.

        self.E_potential = E_potential
        self.E_kinetic = E_kinetic
        self.E_total = E_total

        self.name = name


class EulerMethod:
    """Class for evaluation of the Euler method of integration.
    Determines new trajectory using discrete timesteps and forces from influencing bodies."""

    def __init__(self, time_step, bodies_list):
        self.time_step = time_step
        self.bodies = bodies_list

    def perform_integration(self):
        pos_arr = np.array([(i.position.x, i.position.y, i.position.z) for i in self.bodies[:]])
        vel_arr = np.array([(i.velocity.x, i.velocity.y, i.velocity.z) for i in self.bodies[:]])
        mass_arr = np.array([[i.mass] for i in self.bodies[:]])
        time_step_arr = np.array([[self.time_step] for i in self.bodies[:]])

        dr_arr = pos_arr[np.newaxis, :] - pos_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr, axis=-1))
        # [body_index][index] = # np.sqrt(np.sum((pos_arr[:, np.newaxis] - pos_arr[np.newaxis, :]) ** 2, axis=-1))
        r_between_arr[~np.isfinite(r_between_arr)] = 1
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        acc_arr = (dr_arr.T * tmp_arr).T
        acc_arr = acc_arr.sum(axis=1)

        pos_arr += vel_arr * time_step_arr

        vel_arr += acc_arr * time_step_arr

        for index, body in enumerate(self.bodies):
            body.position.x = pos_arr[index][0]
            body.position.y = pos_arr[index][1]
            body.position.z = pos_arr[index][2]

            body.velocity.x = vel_arr[index][0]
            body.velocity.y = vel_arr[index][1]
            body.velocity.z = vel_arr[index][2]


class EulerCromerMethod:
    """Class for evaluation of the Euler-Cromer method of integration.
    Symplectic version of Euler method, updating velocity from acting forces then position instead."""

    def __init__(self, time_step, bodies_list):
        self.time_step = time_step
        self.bodies = bodies_list

    def perform_integration(self):
        pos_arr = np.array([(i.position.x, i.position.y, i.position.z) for i in self.bodies[:]])
        vel_arr = np.array([(i.velocity.x, i.velocity.y, i.velocity.z) for i in self.bodies[:]])
        mass_arr = np.array([[i.mass] for i in self.bodies[:]])
        time_step_arr = np.array([[self.time_step] for i in self.bodies[:]])

        dr_arr = pos_arr[np.newaxis, :] - pos_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr, axis=-1))
        r_between_arr[~np.isfinite(r_between_arr)] = 1
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        acc_arr = (dr_arr.T * tmp_arr).T
        acc_arr = acc_arr.sum(axis=1)

        vel_arr += acc_arr * time_step_arr

        pos_arr += vel_arr * time_step_arr

        for index, body in enumerate(self.bodies):
            body.position.x = pos_arr[index][0]
            body.position.y = pos_arr[index][1]
            body.position.z = pos_arr[index][2]

            body.velocity.x = vel_arr[index][0]
            body.velocity.y = vel_arr[index][1]
            body.velocity.z = vel_arr[index][2]


class EulerRichardsonMethod:
    """Class for evaluation of the Euler method of integration.
    Midstep version of Euler method."""

    def __init__(self, time_step, bodies_list):
        self.time_step = time_step
        self.bodies = bodies_list

    def perform_integration(self):
        pos_arr = np.array([(i.position.x, i.position.y, i.position.z) for i in self.bodies[:]])
        vel_arr = np.array([(i.velocity.x, i.velocity.y, i.velocity.z) for i in self.bodies[:]])
        mass_arr = np.array([[i.mass] for i in self.bodies[:]])
        time_step_arr = np.array([[self.time_step] for i in self.bodies[:]])

        dr_arr = pos_arr[np.newaxis, :] - pos_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr, axis=-1))
        r_between_arr[~np.isfinite(r_between_arr)] = 1
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        start_acc_arr = (dr_arr.T * tmp_arr).T
        # start_acc_arr[~np.isfinite(start_acc_arr)] = 0
        start_acc_arr = start_acc_arr.sum(axis=1)

        vel_mid_arr = vel_arr + start_acc_arr * 0.5 * time_step_arr

        pos_mid_arr = pos_arr + vel_arr * 0.5 * time_step_arr

        dr_arr = pos_mid_arr[np.newaxis, :] - pos_mid_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr, axis=-1))
        r_between_arr[~np.isfinite(r_between_arr)] = 1
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        acc_mid_arr = (dr_arr.T * tmp_arr).T
        # start_acc_arr[~np.isfinite(start_acc_arr)] = 0
        acc_mid_arr = acc_mid_arr.sum(axis=1)

        vel_arr += acc_mid_arr * time_step_arr

        pos_arr += vel_mid_arr * time_step_arr

        for index, body in enumerate(self.bodies):
            body.position.x = pos_arr[index][0]
            body.position.y = pos_arr[index][1]
            body.position.z = pos_arr[index][2]

            body.velocity.x = vel_arr[index][0]
            body.velocity.y = vel_arr[index][1]
            body.velocity.z = vel_arr[index][2]


class RK4Method:
    """Class for evaluation of the Fourth Order Runge-Kutta method of integration.
    Takes an average of the forces acting over a timestep using different starting forces."""
    # TODO: rewrite as matrices

    def __init__(self, time_step, bodies_list):
        self.time_step = time_step
        self.bodies = bodies_list

    def __partial_step(self, p1, p2, time_step):
        ret = VectorData((0,0,0))
        ret.x = p1.x + p2.x * time_step
        ret.y = p1.y + p2.y * time_step
        ret.z = p1.z + p2.z * time_step
        return ret

    def acc_func(self, other_body, focus_body_current, vec):
        dx = (other_body.position.x - focus_body_current.x)
        dy = (other_body.position.y - focus_body_current.y)
        dz = (other_body.position.z - focus_body_current.z)

        r_between = ((dx * dx) + (dy * dy) + (dz * dz))
        r_between = mp.sqrt(r_between)
        tmp = G_const * other_body.mass / (r_between * r_between * r_between)

        vec.x += tmp * dx
        vec.y += tmp * dy
        vec.z += tmp * dz

        return vec

    def calc_trajectory(self, body_index):
        focus_body = self.bodies[body_index]

        k1v = VectorData((0, 0, 0))
        k2v = VectorData((0, 0, 0))
        k3v = VectorData((0, 0, 0))
        k4v = VectorData((0, 0, 0))
        k1p = VectorData((0, 0, 0))
        k2p = VectorData((0, 0, 0))
        k3p = VectorData((0, 0, 0))
        k4p = VectorData((0, 0, 0))

        for index, other_body in enumerate(self.bodies):
            if index != body_index:
                k1v = self.acc_func(other_body, focus_body.position, k1v)

                k1p.x = focus_body.velocity.x
                k1p.y = focus_body.velocity.y
                k1p.z = focus_body.velocity.z

                k2dp = self.__partial_step(focus_body.position, k1p, (self.time_step * 0.5))
                k2v = self.acc_func(other_body, k2dp, k2v)
                k2p = self.__partial_step(focus_body.velocity, k1v, (self.time_step * 0.5))

                k3dp = self.__partial_step(focus_body.position, k2p, (self.time_step * 0.5))
                k3v = self.acc_func(other_body, k3dp, k3v)
                k3p = self.__partial_step(focus_body.velocity, k2v, (self.time_step * 0.5))

                k4dp = self.__partial_step(focus_body.position, k3p, self.time_step)
                k4v = self.acc_func(other_body, k4dp, k4v)
                k4p = self.__partial_step(focus_body.velocity, k3v, self.time_step)

        focus_body.velocity.x = (focus_body.velocity.x + (self.time_step / 6.) *
                                 (k1v.x + (2. * k2v.x) + (2. * k3v.x) + k4v.x))
        focus_body.velocity.y = (focus_body.velocity.y + (self.time_step / 6.) *
                                 (k1v.y + (2. * k2v.y) + (2. * k3v.y) + k4v.y))
        focus_body.velocity.z = (focus_body.velocity.z + (self.time_step / 6.) *
                                 (k1v.z + (2. * k2v.z) + (2. * k3v.z) + k4v.z))

        focus_body.position.x = (focus_body.position.x + (self.time_step / 6.) *
                                 (k1p.x + (2. * k2p.x) + (2. * k3p.x) + k4p.x))
        focus_body.position.y = (focus_body.position.y + (self.time_step / 6.) *
                                 (k1p.y + (2. * k2p.y) + (2. * k3p.y) + k4p.y))
        focus_body.position.z = (focus_body.position.z + (self.time_step / 6.) *
                                 (k1p.z + (2. * k2p.z) + (2. * k3p.z) + k4p.z))

    def perform_integration(self):
        for body_index, focus_body in enumerate(self.bodies):
            self.calc_trajectory(body_index)


class VelocityVerletMethod: # synchronised / integer step version of leapfrog
    """Class for evaluation of the Velocity Verlet method of integration.
    Updates position and velocity at the same time variable unlike Leapfrog,
    and incorporates velocity solving first time step problem in basic Verlet method.."""

    def __init__(self, time_step, bodies_list):
        self.time_step = time_step
        self.bodies = bodies_list

    def perform_integration(self):
        pos_arr = np.array([(i.position.x, i.position.y, i.position.z) for i in self.bodies[:]])
        vel_arr = np.array([(i.velocity.x, i.velocity.y, i.velocity.z) for i in self.bodies[:]])
        mass_arr = np.array([[i.mass] for i in self.bodies[:]])

        dr_arr = pos_arr[np.newaxis, :] - pos_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr, axis=-1))
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        start_acc_arr = (dr_arr.T * tmp_arr).T
        # start_acc_arr[~np.isfinite(start_acc_arr)] = 0
        start_acc_arr = start_acc_arr.sum(axis=1)

        pos_arr += vel_arr * self.time_step + 0.5 * start_acc_arr * self.time_step * self.time_step

        dr_arr = pos_arr[np.newaxis, :] - pos_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr, axis=-1))
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        new_acc_arr = (dr_arr.T * tmp_arr).T
        # new_acc_arr[~np.isfinite(new_acc_arr)] = 0
        new_acc_arr = new_acc_arr.sum(axis=1)

        vel_arr += (start_acc_arr + new_acc_arr) * 0.5 * self.time_step

        for index, body in enumerate(self.bodies):
            body.position.x = pos_arr[index][0]
            body.position.y = pos_arr[index][1]
            body.position.z = pos_arr[index][2]

            body.velocity.x = vel_arr[index][0]
            body.velocity.y = vel_arr[index][1]
            body.velocity.z = vel_arr[index][2]


# class LeapfrogMethod:
#     """Class for evaluation of the Leapfrog method of integration.
#     Updates position and velocity at staggered time steps and 'leapfrog' over each other."""
# Not needed as variable leapfrog covers all analysis points


class VariableLeapfrogMethod:
    """"""

    def __init__(self, time_step, bodies_list):
        self.time_step = time_step # Use largest timestep
        self.bodies = bodies_list
        self.time = 0.
        self.fixed_ts = False

    def _calculate_acceleration(self, body_index):
        """Calculates the net acceleration of a body at the beginning of the step
        from all gravitational forces acting on it."""

        acceleration = VectorData((0, 0, 0))
        focus_body = self.bodies[body_index]

        for index, other_body in enumerate(self.bodies):
            if index != body_index:
                dx = (other_body.old_position.x - focus_body.position.x)
                dy = (other_body.old_position.y - focus_body.position.y)
                dz = (other_body.old_position.z - focus_body.position.z)

                r_between = ((dx * dx) + (dy * dy) + (dz * dz))
                r_between = mp.sqrt(r_between)
                tmp = G_const * other_body.mass / (r_between * r_between * r_between)
                acceleration.x += tmp * dx
                acceleration.y += tmp * dy
                acceleration.z += tmp * dz

        return acceleration

    def perform_integration(self):
        self.time += self.time_step

        for body_index, focus_body in enumerate(self.bodies):
            focus_body.old_position = focus_body.position

        for body_index, focus_body in enumerate(self.bodies):
            if self.fixed_ts is False and self.time_step % focus_body.time_step != 0:
                # corrects timesteps that would step out of range, designed for precision by choosing smaller timestep
                focus_body.time_step = self.time_step / int(math.ceil(self.time_step / focus_body.time_step))
                # print("changing " + str(focus_body.name) + " timestep to " + str(focus_body.time_step))

            while focus_body.body_time < self.time:
                acceleration = self._calculate_acceleration(body_index)
                focus_body.velocity.x += acceleration.x * focus_body.time_step * 0.5
                focus_body.velocity.y += acceleration.y * focus_body.time_step * 0.5
                focus_body.velocity.z += acceleration.z * focus_body.time_step * 0.5

                focus_body.position.x += focus_body.velocity.x * focus_body.time_step
                focus_body.position.y += focus_body.velocity.y * focus_body.time_step
                focus_body.position.z += focus_body.velocity.z * focus_body.time_step

                acceleration = self._calculate_acceleration(body_index)
                focus_body.velocity.x += acceleration.x * focus_body.time_step * 0.5
                focus_body.velocity.y += acceleration.y * focus_body.time_step * 0.5
                focus_body.velocity.z += acceleration.z * focus_body.time_step * 0.5

                focus_body.body_time += focus_body.time_step
        self.fixed_ts = True


class VariableLeapfrogMethodM:  # Fixed timestep because of matrices
    """"""

    def __init__(self, time_step, bodies_list):  # Bodies with.position, mass,.velocity, name data
        self.time_step = time_step  # Change for variable timestep per body
        self.bodies = bodies_list

    def perform_integration(self):
        pos_arr = np.array([(i.position.x, i.position.y, i.position.z) for i in self.bodies[:]])
        vel_arr = np.array([(i.velocity.x, i.velocity.y, i.velocity.z) for i in self.bodies[:]])
        mass_arr = np.array([[i.mass] for i in self.bodies[:]])
        time_step_arr = np.array([[self.time_step] for i in self.bodies[:]])

        dr_arr = pos_arr[np.newaxis, :] - pos_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr,
                                       axis=-1))

        r_between_arr[~np.isfinite(r_between_arr)] = 1
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        start_acc_arr = (dr_arr.T * tmp_arr).T  # tmp_arr.T * dr_arr
        # start_acc_arr[~np.isfinite(start_acc_arr)] = 0
        start_acc_arr = start_acc_arr.sum(axis=1)

        vel_mid_arr = vel_arr + start_acc_arr * 0.5 * time_step_arr  # Still fixed

        pos_arr += vel_mid_arr * time_step_arr

        # Variable Leapfrog method 'kick-drift-kick' # Pos, vel, pos may also work i believe?
        dr_arr = pos_arr[np.newaxis, :] - pos_arr[:, np.newaxis]
        r_between_arr = np.sqrt(np.sum(dr_arr * dr_arr,
                                       axis=-1))
        tmp_arr = G_const * mass_arr / (r_between_arr * r_between_arr * r_between_arr)
        tmp_arr[~np.isfinite(tmp_arr)] = 0
        new_acc_arr = (dr_arr.T * tmp_arr).T  # tmp_arr.T * dr_arr
        # new_acc_arr[~np.isfinite(new_acc_arr)] = 0
        new_acc_arr = new_acc_arr.sum(axis=1)

        vel_arr = vel_mid_arr + new_acc_arr * 0.5 * time_step_arr

        for index, body in enumerate(self.bodies):
            body.position.x = pos_arr[index][0]
            body.position.y = pos_arr[index][1]
            body.position.z = pos_arr[index][2]

            body.velocity.x = vel_arr[index][0]
            body.velocity.y = vel_arr[index][1]
            body.velocity.z = vel_arr[index][2]
