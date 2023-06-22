import rllib

from typing import List
import numpy as np
import copy

from universe.common import transform2state, Transform
from universe.common import get_leading_vehicle
from universe.common import Vehicle, EndToEndVehicle

from .agents_master import AgentListMaster



class IdmVehicleAsNeural(EndToEndVehicle):
    def __init__(self, config, vi, global_path, transform, character):
        super().__init__(config, vi, global_path, transform)
        self.character = character

        self.leading_range = 50.0
        self.idm_scale_x = config.idm_scale_x
        self.idm_scale_y = config.idm_scale_y
        assert self.leading_range <= self.perception_range

        self.desired_speed = self.max_velocity *2
        self.time_gap = 1.0  ## 1.6
        self.min_gap = 2.0
        self.delta = 4.0
        self.acceleration = 1.0
        self.acceleration = self.max_acceleration  ### !warning
        self.comfortable_deceleration = 1.5  ## 1.7
        self.comfortable_deceleration = -self.min_acceleration  ### !warning


    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        v = str(round(state.v, 2))
        leading_vi = None
        if hasattr(self, 'leading_vehicle') and self.leading_vehicle != None:
            leading_vi = self.leading_vehicle.vi
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, v: {v}, leading: {leading_vi}>'



    def get_target(self, reference: List[Vehicle]):
        vehicles = reference

        '''get leading vehicle'''
        current_transform = self.get_transform()
        remaining_transforms, remaining_distance = self.global_path.remaining_transforms(current_transform)
        vehicle, transform, distance = get_leading_vehicle(self, vehicles, remaining_transforms, max_distance=self.leading_range, scale_x=self.idm_scale_x, scale_y=self.idm_scale_y)
        self.leading_vehicle = vehicle

        target_v = self.desired_speed
        if vehicle != None:
            target_v = self.intelligent_driver_model(vehicle, transform, distance)
        return target_v


    def get_control(self, target):
        current_transform = self.get_transform()
        target_transform, curvature = self.global_path.target_transform(current_transform)

        current_state = self.get_state()
        target_state = transform2state(target_transform, v=target, k=curvature)
        control = self.controller.run_step(current_state, target_state)
        return control


    def intelligent_driver_model(self, leading_vehicle: Vehicle, leading_transform: Transform, leading_distance):
        distance_c2c = leading_distance
        length_two_half = leading_vehicle.bbx_x + self.bbx_x
        distance_b2b = distance_c2c - length_two_half - 0.3   # bumper-to-bumper distance
        distance_b2b_valid = max(0.001, distance_b2b)

        leading_v = leading_vehicle.get_state().v
        leading_v *= np.cos(leading_vehicle.get_state().theta - leading_transform.theta)
        current_v = self.get_state().v
        delta_v = current_v - leading_v
        
        s = current_v*(self.time_gap+delta_v/(2*np.sqrt(self.acceleration*self.comfortable_deceleration)))
        distance_desired = self.min_gap + max(0, s)

        v_rational = (current_v / self.desired_speed)**self.delta
        s_rational = (distance_desired / distance_b2b_valid) ** 2
        acceleration_target = self.acceleration * (1 - v_rational - s_rational)
        target_v = current_v + acceleration_target / self.decision_frequency


        # a = (distance_b2b, (self.min_gap + current_v*self.time_gap) / (1-(current_v/self.desired_speed)**self.delta)*0.5)
        # print(a, (target_v, current_v), (acceleration_target, 1-v_rational, s_rational) )
        # print()

        return target_v








class AgentListMasterNormalBackground(AgentListMaster):
    """
        Only for multi agent.
    """
    

    def observe(self, step_reset, time_step):
        # if time_step == 0:
        #     for vehicle in self.vehicles_neural + self.vehicles_rule:
        #         self.vehicle_states[vehicle.vi, :self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
        #         self.vehicle_masks[vehicle.vi, :self.horizon-1] = 1
        
        # for vehicle in self.vehicles_neural + self.vehicles_rule:
        #     self.vehicle_states[vehicle.vi, time_step + self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
        #     self.vehicle_masks[vehicle.vi, time_step + self.horizon-1] = 3

        # vehicle_states = self.vehicle_states[:, time_step:time_step+self.horizon]
        # vehicle_masks = self.vehicle_masks[:, time_step:time_step+self.horizon]

        # state = self.perception.run_step(step_reset, time_step, self.vehicles_neural + self.vehicles_rule, vehicle_states, vehicle_masks)
        # self.state_neural_nackground = state[1:]
        return [rllib.basic.Data(step_reset=step_reset, time_step=time_step, vi=agent.vi, bound_flag=np.array([False])) for agent in self.vehicles_neural]




    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """

        assert len(references) == len(self.vehicles_neural)

        # import pdb; pdb.set_trace()

        vehicles = self.vehicles_neural + self.vehicles_rule
        targets_neural = [vehicle.get_target(vehicles) for vehicle in self.vehicles_neural]
        targets = targets_neural

        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return



