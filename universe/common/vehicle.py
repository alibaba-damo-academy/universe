
import numpy as np
from enum import Enum
from abc import ABC
from typing import List

from .geo import State, transform2state, state2transform
from .actor import ActorBoundingBox
from .color import ColorLib
from .global_path import GlobalPath
from .vehicle_model import BicycleModel
from .vehicle_controller import Controller



class VehicleType(Enum):
    static = -1
    neural = 0
    idm = 1




class Vehicle(object):
    def __init__(self, config, vi, global_path, transform):
        self.config = config
        self.vi = vi
        self.global_path: GlobalPath = global_path

        self.decision_frequency = config.decision_frequency
        self.control_frequency = config.control_frequency
        self.perception_range = config.perception_range

        self.bbx_x = config.bbx_x
        self.bbx_y = config.bbx_y

        self.decision_dt, self.control_dt = 1.0/self.decision_frequency, 1.0/self.control_frequency
        self.min_velocity = config.min_velocity
        self.max_velocity = config.max_velocity
        self.max_acceleration = config.max_acceleration
        self.min_acceleration = config.min_acceleration
        self.max_steer = config.max_steer
        self.wheelbase = config.wheelbase
        self.max_curvature = np.tan(self.max_steer) / self.wheelbase

        self.transform = transform
        self.state = transform2state(transform, v=0.0)
        self.historical_states: List[State] = []
        self.future_states: List[State] = []

        self.vehicle_model = BicycleModel(config, self.control_dt)
        self.controller = Controller(config, self.control_dt, self.wheelbase)
        return


    def __str__(self):
        return str(type(self))[:-1] + f' vi: {self.vi}>'

    def __repr__(self):
        return str(self)


    def get_transform(self):
        return self.transform
    
    def get_state(self):
        return self.state


    @property
    def bounding_box(self):
        return ActorBoundingBox(self.transform, self.bbx_x, self.bbx_y)



    def get_target(self, reference, **kwargs):
        return self.max_velocity
    

    def get_control(self, target):
        current_transform = self.get_transform()
        target_transform, curvature = self.global_path.target_transform(current_transform)

        current_state = self.get_state()
        target_state = transform2state(target_transform, v=target, k=curvature)
        control = self.controller.run_step(current_state, target_state)
        return control



    def tick(self):
        self.historical_states.insert(0, self.state)


    def forward(self, control):
        next_state = self.vehicle_model(self.state, control)
        self.state = next_state
        self.transform = state2transform(next_state)

    def set_state(self, next_state):
        self.state = next_state
        self.transform = state2transform(next_state)



    def reach_goal(self, preview_distance=0):
        return self.global_path.reached(preview_distance)



    def render(self, ax, color=None):
        for i, state in enumerate(self.historical_states[:10:1]):
            ActorBoundingBox(state, self.bbx_x, self.bbx_y).render(ax, color, alpha=np.exp(-(i+1)*0.3))
        patch = ActorBoundingBox(self.state, self.bbx_x, self.bbx_y).render(ax, color, label=str(self), alpha=1.0)
        line = ActorBoundingBox(self.state, self.bbx_x, self.bbx_y).render_polyline(ax, color=ColorLib.normal(ColorLib.black))

        lines = [line]
        future_x = [s.x for s in self.future_states]
        future_y = [s.y for s in self.future_states]
        if len(future_x) > 0:
            line1 = ax.plot(future_x, future_y, '-', color=color, zorder=3)[0]
            line2 = ax.plot(future_x[-1], future_y[-1], 'o', color=color, linewidth=2, zorder=3)[0]
            lines.append(line1)
            lines.append(line2)
        
        ### optional
        # text = ax.text(self.state.x, self.state.y, str(self.vi), weight='bold', color=ColorLib.normal(ColorLib.black), zorder=9)
        # lines.append(text)
        return lines, patch

