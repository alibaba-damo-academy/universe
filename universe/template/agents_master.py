
import numpy as np
import copy

from ..common import Vehicle, VehicleControl
from ..common import EndToEndVehicle
from ..common import IdmVehicle, RuleVehicle
from .. import AgentsMaster




class EndToEndVehicleWithCharacter(EndToEndVehicle):
    def __init__(self, config, vi, global_path, transform, character):
        super().__init__(config, vi, global_path, transform)
        self.character = character

    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        character = str(np.round(self.character, 2))
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, svo: {character}>'

class IdmVehicleWithCharacter(IdmVehicle):
    def __init__(self, config, vi, global_path, transform, character):
        super().__init__(config, vi, global_path, transform)
        self.character = character

    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        v = str(round(state.v, 2))
        leading_vi = None
        if hasattr(self, 'leading_vehicle') and self.leading_vehicle != None:
            leading_vi = self.leading_vehicle.vi
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, v: {v}, leading: {leading_vi}>'





class EndToEndVehicleWithCharacterBackground(RuleVehicle):
    dim_action = 2

    def __init__(self, config, vi, global_path, transform, character):
        super().__init__(config, vi, global_path, transform)
        self.character = character

    def set_svo(self, reference):
        character = np.clip(reference[0], -1,1)
        self.character = ((character + 1) *0.5).astype(np.float32)
        return


    def get_target(self, reference):
        acc = np.clip(reference[0].item(), -1,1) * self.max_acceleration
        steer = np.clip(reference[1].item(), -1,1) * self.max_steer
        return VehicleControl(acc, steer)


    def get_control(self, target):
        control = target
        return control


    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        v = str(round(state.v, 2))
        character = str(np.round(self.character, 2))
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, v: {v}, svo: {character}>'









class AgentListMaster(AgentsMaster):
    class GetVehicleState(object):
        dim_state = 5
        def run_step(self, vehicle: Vehicle):
            state = vehicle.get_state()
            return np.array([
                state.x, state.y, state.theta, state.v,
                vehicle.character,
            ], dtype=np.float32)




