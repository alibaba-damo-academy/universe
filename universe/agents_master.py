import rldev

import numpy as np
from typing import List


from .common.color import ColorLib
from .common.topology_map import TopologyMap
from .common.vehicle import Vehicle
from .common.vehicle_rule import RuleVehicle, IdmVehicle
from .common.vehicle_neural import NeuralVehicle, EndToEndVehicle

from .perception import Perception


class AgentsMaster(object):
    class GetVehicleState(object):
        dim_state = 4
        def run_step(self, vehicle: Vehicle):
            state = vehicle.get_state()
            return np.array([
                state.x, state.y, state.theta, state.v,
            ], dtype=np.float32)
    
    horizon = 10
    
    def __init__(self, config: rldev.YamlConfig, topology_map: TopologyMap, **kwargs):
        self.config = config

        self.decision_frequency = config.decision_frequency
        self.control_frequency = config.control_frequency
        self.skip_num = int(self.control_frequency // self.decision_frequency)
        assert self.control_frequency % self.decision_frequency == 0

        self.perception_range = config.perception_range

        self.vehicles_rule: List[RuleVehicle] = []
        self.vehicles_neural: List[NeuralVehicle] = []

        self.get_vehicle_state = self.GetVehicleState()
        self.dim_vehicle_state = self.get_vehicle_state.dim_state

        self.neural_vehicle_cls = config.get('neural_vehicle_cls', EndToEndVehicle)
        self.rule_vehicle_cls = config.get('rule_vehicle_cls', IdmVehicle)

        perception_cls = config.get('perception_cls', Perception)
        self.perception = perception_cls(config, topology_map, self.dim_vehicle_state, self.horizon)
        self.dim_state = perception_cls.dim_state
        self.dim_action = self.neural_vehicle_cls.dim_action
        return


    def reset(self, num_steps):
        self.num_steps = num_steps
        self.vehicle_states = np.ones((0, self.num_steps + self.horizon-1, self.dim_vehicle_state), dtype=np.float32) *np.inf
        self.vehicle_masks = np.zeros((0, self.num_steps + self.horizon-1), dtype=np.int)
        self._vehicle_state = np.ones((1, self.num_steps + self.horizon-1, self.dim_vehicle_state), dtype=np.float32) *np.inf
        self._vehicle_mask = np.zeros((1, self.num_steps + self.horizon-1), dtype=np.int)
        return


    def register_vehicle(self, vehicle: Vehicle):
        if isinstance(vehicle, RuleVehicle):
            self.vehicles_rule.append(vehicle)
        if isinstance(vehicle, NeuralVehicle):
            self.vehicles_neural.append(vehicle)
        if type(vehicle) == Vehicle:
            raise RuntimeError('Please use explicit rule or neural based vehicle')
        self.vehicle_states = np.concatenate([self.vehicle_states, self._vehicle_state], axis=0)
        self.vehicle_masks = np.concatenate([self.vehicle_masks, self._vehicle_mask], axis=0)
        return


    def has(self, vehicle: Vehicle):
        vis = [v.vi for v in self.vehicles_rule + self.vehicles_neural]
        return vehicle.vi in vis
    
    def remove(self, vehicle):
        if vehicle in self.vehicles_rule:
            self.vehicles_rule.remove(vehicle)
        if vehicle in self.vehicles_neural:
            self.vehicles_neural.remove(vehicle)
        return
    
    
    def destroy(self):
        self.vehicles_rule, self.vehicles_neural = [], []
        self.vehicle_states = None
        self.vehicle_masks = None
        return
    

    def get_vehicle_cls(self, num_agents):
        if len(self.vehicles_neural) < num_agents:
            vehicle_cls = self.neural_vehicle_cls
        else:
            vehicle_cls = self.rule_vehicle_cls
        return vehicle_cls




    def observe(self, step_reset, time_step):
        if time_step == 0:
            for vehicle in self.vehicles_neural + self.vehicles_rule:
                self.vehicle_states[vehicle.vi, :self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
                self.vehicle_masks[vehicle.vi, :self.horizon-1] = 1
        
        for vehicle in self.vehicles_neural + self.vehicles_rule:
            self.vehicle_states[vehicle.vi, time_step + self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
            self.vehicle_masks[vehicle.vi, time_step + self.horizon-1] = 3

        vehicle_states = self.vehicle_states[:, time_step:time_step+self.horizon]
        vehicle_masks = self.vehicle_masks[:, time_step:time_step+self.horizon]

        state = self.perception.run_step(step_reset, time_step, self.vehicles_neural, vehicle_states, vehicle_masks)
        return state




    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """

        assert len(references) == len(self.vehicles_neural)

        vehicles = self.vehicles_neural + self.vehicles_rule
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(vehicles) for vehicle in self.vehicles_rule]
        targets = targets_neural + targets_rule

        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return
    


    def render(self, ax):
        lines, patches = [], []
        colors = list(ColorLib)[:8]
        for vehicle in self.vehicles_neural:
            line, patch = vehicle.render(ax, ColorLib.normal(colors[vehicle.vi %len(colors)]))
            lines.extend(line)
            patches.append(patch)
        for vehicle in self.vehicles_rule:
            color = ColorLib.grey
            line, patch = vehicle.render(ax, ColorLib.normal(color))
            lines.extend(line)
            patches.append(patch)
        return lines, patches

