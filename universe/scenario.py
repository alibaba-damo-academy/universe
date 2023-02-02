import rldev

import numpy as np
from typing import List, Dict
import random
import copy

from .common.geo import Vector, Transform, error_transform
from .common.color import ColorLib
from .common.global_path import GlobalPath
from .common.topology_map import TopologyMap
from .common.vehicle import Vehicle
from .agents_master import AgentsMaster
from .scenario_randomization import ScenarioRandomization



class Scenario(object):
    def __init__(self, config: rldev.YamlConfig, env_index, centerline, sideline):
        self.config = config
        self.env_index = env_index

        self.boundary = config.boundary
        self.center = config.center
        self.x_range, self.y_range = self.boundary.x_max-self.boundary.x_min, self.boundary.y_max-self.boundary.y_min

        topology_map_cls = config.get('topology_map', TopologyMap)
        centerline, sideline = self.clip_topology(centerline, sideline)
        self.topology_map = topology_map_cls(centerline, sideline)

        self.num_vehicles_range = config.num_vehicles_range
        self.num_vehicles_min = self.num_vehicles_range.min
        self.num_vehicles_max = self.num_vehicles_range.max

        self.generate_global_paths()
        self.generate_spawn_transforms()


    def clip_topology(self, centerline, sideline):
        """
            centerline: shape is (num_lines, num_points, num_features)
        """

        centerline_x, centerline_y = centerline[...,0], centerline[...,1]
        sideline_x, sideline_y = sideline[...,0], sideline[...,1]

        centerline_mask = np.expand_dims(np.where(
            (centerline_x <= self.boundary.x_max) &
            (centerline_x >= self.boundary.x_min) &
            (centerline_y <= self.boundary.y_max) &
            (centerline_y >= self.boundary.y_min)
        ,1, 0), axis=-1).repeat(centerline.shape[-1], axis=2)
        centerline_clip = np.where(centerline_mask == 1, centerline, np.inf)
        centerline_clip = np.delete(centerline_clip, np.where(centerline_mask[...,0].sum(axis=1) == 0)[0], axis=0)

        sideline_mask = np.expand_dims(np.where(
            (sideline_x <= self.boundary.x_max) &
            (sideline_x >= self.boundary.x_min) &
            (sideline_y <= self.boundary.y_max) &
            (sideline_y >= self.boundary.y_min)
        ,1, 0), axis=-1).repeat(sideline.shape[-1], axis=2)
        sideline_clip = np.where(sideline_mask == 1, sideline, np.inf)
        sideline_clip = np.delete(sideline_clip, np.where(sideline_mask[...,0].sum(axis=1) == 0)[0], axis=0)

        def left_align(line):
            mask = np.where(line < np.inf, 1, 0)
            max_valid_length = mask[...,0].sum(axis=1).max()
            sorted_index = np.argsort(-mask, axis=1, kind='mergesort')
            return np.take_along_axis(line, sorted_index, axis=1)[:,:max_valid_length].copy()

        return left_align(centerline_clip), left_align(sideline_clip)


    def generate_global_paths(self):
        self.global_paths: List[GlobalPath] = []
        return

    def generate_spawn_transforms(self):
        self.spawn_transforms: Dict[Transform, GlobalPath] = {}
        self.special_spawn_transforms: Dict[Transform, GlobalPath] = {}
        return




    def reset(self, step_reset, sa=True):
        self.step_reset = step_reset
        self.num_vehicles = random.randint(self.num_vehicles_min, self.num_vehicles_max)
        if sa:
            self.num_agents = 1
        else:
            self.num_agents = self.num_vehicles
        self.generate_scenario_randomization()
        if self.num_vehicles < self.scenario_randomization.num_vehicles:
            self.scenario_randomization.num_vehicles = self.num_vehicles
        return
    

    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        self.bbx_vectors = [Vector(x=self.config.bbx_x, y=self.config.bbx_y)] *len(self.spawn_transforms)
        self.scenario_randomization = scenario_randomization_cls(self.spawn_transforms, self.bbx_vectors, self.num_vehicles)
        return



    def register_agents(self, agents_master: AgentsMaster):
        for vi in range(self.scenario_randomization.num_vehicles):
            config_vehicle = copy.copy(self.config.config_vehicle)
            config_vehicle.set('decision_frequency', self.config.decision_frequency)
            config_vehicle.set('control_frequency', self.config.control_frequency)
            config_vehicle.set('perception_range', self.config.perception_range)
            vehicle_cls = agents_master.get_vehicle_cls(num_agents=self.num_agents)
            vehicle = vehicle_cls(config_vehicle, **self.scenario_randomization[vi].to_dict())
            agents_master.register_vehicle(vehicle)
        return


    def at_junction(self, vehicle: Vehicle):
        return False


    def wrong_lane(self, vehicle: Vehicle):
        t = vehicle.get_transform()
        centerline_x, centerline_y = self.topology_map.centerline[...,0], self.topology_map.centerline[...,1]
        dist = np.sqrt((centerline_x-t.x)**2 + (centerline_y-t.y)**2)

        index = np.unravel_index(np.argmin(dist), dist.shape)
        closest_x, closest_y = self.topology_map.centerline[index][0], self.topology_map.centerline[index][1]
        closest_theta = self.topology_map.centerline_theta[index]
        closest_t = Transform(closest_x, closest_y, closest_theta)

        _, _, e_theta = error_transform(t, closest_t)
        wrong_lane = abs(np.rad2deg(rldev.pi2pi_numpy(e_theta))) > 90
        return wrong_lane


    def in_boundary(self, vehicle: Vehicle):
        t = vehicle.get_transform()
        x, y = t.x, t.y
        flag = False
        if x >= self.boundary.x_min and x <= self.boundary.x_max and y >= self.boundary.y_min and y <= self.boundary.y_max:
            flag = True
        return flag


    def finish_task(self, vehicle: Vehicle):
        return True


    def render(self, ax):
        ax.set_xlim(self.boundary.x_min, self.boundary.x_max)
        ax.set_ylim(self.boundary.y_min, self.boundary.y_max)
        self.topology_map.render(ax)

        ### global_path, optional
        # for global_path in self.global_paths:
        #     global_path.render(ax, length=1.0, width=0.05, linewidth=2.0)

        ### spawn_transform, optional
        # for spawn_transform in self.spawn_transforms:
        #     spawn_transform.render(ax, length=1.3, width=0.10, linewidth=2.5)

        ### special_spawn_transform, optional
        # for spawn_transform in self.special_spawn_transforms:
        #     spawn_transform.render(ax, length=1.3, width=0.10, linewidth=2.5)
        
        ### bbx, optional
        # for bbx, color in zip(self.bbxs, list(ColorLib)):
        #     print(bbx, color)
        #     bbx.render(ax, ColorLib.normal(color))
        return


