
import numpy as np
from typing import List


from .geo import Vector, error_transform
from .vehicle_model import VehicleControl
from .vehicle import Vehicle


class NeuralVehicle(Vehicle):
    def __init__(self, config, vi, global_path, transform):
        super().__init__(config, vi, global_path, transform)





class EndToEndVehicle(NeuralVehicle):
    dim_action = 2
    
    def get_target(self, reference):
        acc = np.clip(reference[0].item(), -1,1) * self.max_acceleration
        steer = np.clip(reference[1].item(), -1,1) * self.max_steer
        return VehicleControl(acc, steer)


    def get_control(self, target):
        control = target
        return control


