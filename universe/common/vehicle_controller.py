
import time
import numpy as np

from .geo import error_state
from .vehicle_model import VehicleControl


class Controller(object):
    def __init__(self, config, dt, wheelbase):
        '''parameter'''
        self.max_acceleration = config.max_acceleration
        self.min_acceleration = config.min_acceleration
        self.max_steer = config.max_steer

        Kp, Ki, Kd = 1.00, 0.01, 0.05
        self.v_param = (Kp, Ki, Kd)
        k_theta, k_e = 0.2, 0.1
        self.w_param = (k_theta, k_e)

        self.acc_controller = LongPID(dt, self.min_acceleration, self.max_acceleration)
        self.steer_controller = LatRWPF(wheelbase, self.max_steer)


    def run_step(self, current_state, target_state):
        target_state.v -= 0.01
        acc = self.acc_controller.run_step(current_state, target_state, self.v_param)
        steer = self.steer_controller.run_step(current_state, target_state, self.w_param)
        return VehicleControl(acc=acc, steer=steer)
    




class LongPID(object):
    def __init__(self, dt, min_a, max_a):
        self.dt = dt
        self.min_a, self.max_a = min_a, max_a
        self.last_error = 0
        self.sum_error = 0

    def run_step(self, current_state, target_state, param):
        Kp, Ki, Kd = param[0], param[1], param[2]

        v_current = current_state.v
        v_target = target_state.v
        error = v_target - v_current

        acceleration = Kp * error
        acceleration += Ki * self.sum_error * self.dt
        acceleration += Kd * (error - self.last_error) / self.dt

        self.last_error = error
        self.sum_error += error
        '''eliminate drift'''
        if abs(self.sum_error) > 10:
            self.sum_error = 0.0

        return np.clip(acceleration, self.min_a, self.max_a)


class LatRWPF(object):
    def __init__(self, wheelbase, max_steer):
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_curvature = np.tan(self.max_steer) / self.wheelbase
        self.curvature_factor = 1.0
        self.alpha = 1.8

    def run_step(self, current_state, target_state, param):
        k_theta, k_e = param[0], param[1]

        longitudinal_e, lateral_e, theta_e = error_state(current_state, target_state)
        kr = target_state.k

        c1 = (kr*self.curvature_factor) *np.cos(theta_e)
        c2 = - k_theta *theta_e
        c3 = (k_e*np.exp(-theta_e**2/self.alpha))*lateral_e
        curvature = c1 + c2 + c3

        curvature = np.clip(curvature, -self.max_curvature, self.max_curvature)
        steer = np.arctan(curvature * self.wheelbase)
        return steer / self.max_steer


