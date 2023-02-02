import rldev

import numpy as np

from .geo import State



class VehicleControl(object):
    def __init__(self, acc, steer):
        self.acc = acc
        self.steer = steer



class BicycleModel(object):
    def __init__(self, config, dt):
        self.min_velocity = config.min_velocity
        self.max_velocity = config.max_velocity
        self.max_acceleration = config.max_acceleration
        self.min_acceleration = config.min_acceleration
        self.max_steer = config.max_steer
        self.wheelbase = config.wheelbase
        self.max_curvature = np.tan(self.max_steer) / self.wheelbase
        self.dt = dt
    

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    def forward(self, state: State, action: VehicleControl):
        acc = np.clip(action.acc, self.min_acceleration, self.max_acceleration)
        steer = np.clip(action.steer, -self.max_steer, self.max_steer)

        x, y, theta, v = state.x, state.y, state.theta, state.v
        next_state = State(
            x=x + self.dt *v * np.cos(theta),
            y=y + self.dt *v * np.sin(theta),
            theta=rldev.pi2pi_numpy(theta + self.dt * v * np.tan(steer) / self.wheelbase),
            v=np.clip(v + self.dt *acc, self.min_velocity, self.max_velocity),
        )
        return next_state


