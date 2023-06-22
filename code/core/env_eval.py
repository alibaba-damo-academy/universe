import rllib
import universe

import copy
import random

from universe.common import Vector, ActorBoundingBox
from universe.env_sa_v0 import EnvInteractiveSingleAgent
from universe.env_ma_v0 import EnvInteractiveMultiAgent



class EnvInteractiveSingleAgent_v1(EnvInteractiveSingleAgent):
    def reset_done_agents(self, episode_info):
        res =  super().reset_done_agents(episode_info)

        raise NotImplementedError

        return res






class EnvInteractiveMultiAgent_v1(EnvInteractiveMultiAgent):
    def reset_done_agents(self, episode_info):
        res =  super().reset_done_agents(episode_info)

        if self.time_step % self.config.spawn_interval != 0:
            return res
        
        spawn_transform = random.choice(list(self.scenario.special_spawn_transforms.keys()))
        bbx_vector = Vector(x=self.config.bbx_x, y=self.config.bbx_y)
        bbx = ActorBoundingBox(spawn_transform, bbx_vector.x, bbx_vector.y)

        collision = False
        for vehicle in self.agents_master.vehicles_neural + self.agents_master.vehicles_rule:
            if bbx.intersects(vehicle.bounding_box):
                collision = True
                break
        
        if not collision:
            scenario_randomization_cls = self.config.get('scenario_randomization_cls', universe.ScenarioRandomization)
            spawn_transforms = {spawn_transform: self.scenario.special_spawn_transforms[spawn_transform]}
            bbx_vectors = [bbx_vector]
            scenario_randomization = scenario_randomization_cls(spawn_transforms, bbx_vectors, num_vehicles=1)

            config_vehicle = copy.copy(self.config.config_vehicle)
            config_vehicle.set('decision_frequency', self.config.decision_frequency)
            config_vehicle.set('control_frequency', self.config.control_frequency)
            config_vehicle.set('perception_range', self.config.perception_range)
            vehicle_cls = self.agents_master.get_vehicle_cls(num_agents=100)
            kwargs = scenario_randomization[0]
            kwargs.vi = self.agents_master.vehicle_states.shape[0]
            vehicle = vehicle_cls(config_vehicle, **kwargs.to_dict())
            self.agents_master.register_vehicle(vehicle)

        print(rllib.basic.prefix(self) + f'time {self.time_step} num vehicles: {self.agents_master.vehicle_states.shape[0]}')
        return res



