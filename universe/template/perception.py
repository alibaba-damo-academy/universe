import rldev

import numpy as np

import matplotlib.pyplot as plt
from typing import List


from ..common import TopologyMap, transform_poses, transform_points
from ..common import Vehicle, EndToEndVehicle




class PerceptionPointNet(object):
    invalid_value = np.inf

    dim_state = rldev.BaseData(agent=5, static=4)  ### ! warning todo


    def __init__(self, config, topology_map: TopologyMap, dim_vehicle_state, horizon):
        self.config = config
        self.decision_frequency = config.decision_frequency
        self.perception_range = config.perception_range
        self.horizon = horizon


        # self.num_vehicles = num_vehicles
        self.num_agents_max = self.config.num_vehicles_range.max  ### ! warning


        self.historical_timestamps = -np.array([range(self.horizon)], dtype=np.float32).T[::-1] /self.decision_frequency
        sampling_resolution = 2.0

        self.perp_route = PerceptionVectorizedRoute(config, sampling_resolution=sampling_resolution)
        self.perp_map = PerceptionVectorizedMap(config, topology_map)


        # self.dim_state = rldev.BaseData(agent=dim_vehicle_state, static=self.perp_map.dim_state)


        '''viz'''
        self.default_colors = rldev.Data(
            ego='r', obs='g', route='b',
            lane='#D3D3D3',  ### lightgray
            bound='#800080', ### purple
        )
        return




    def run_step(self, step_reset, time_step, agents: List[EndToEndVehicle], vehicle_states, vehicle_masks):
        self.step_reset, self.time_step = step_reset, time_step

        states = []
        for agent in agents:
            state_vehicle = self.get_vehicles(agent, vehicle_states, vehicle_masks)
            state_route = self.perp_route.run_step(agent)
            state_map = self.perp_map.run_step(step_reset, time_step, agent)

            ### only for multi-agent
            agent_state = agent.get_state()
            agent_states = vehicle_states[:,-1]
            dist = np.sqrt((agent_states[:,0]-agent_state.x)**2 + (agent_states[:,1]-agent_state.y)**2)
            agent_masks = np.where(dist < agent.perception_range, 1, 0)

            state = rldev.Data(step_reset=step_reset, time_step=time_step, vi=agent.vi, 
                agent_masks=agent_masks, vehicle_masks=vehicle_masks[:,-1]) + state_vehicle + state_route + state_map
            states.append(state)
        return states



    def get_vehicles(self, agent: EndToEndVehicle, vehicle_states, vehicle_masks):
        ego_states = vehicle_states[agent.vi]
        ego_masks = vehicle_masks[agent.vi]
        other_states = np.delete(vehicle_states, agent.vi, axis=0)
        other_masks = np.delete(vehicle_masks, agent.vi, axis=0)  ## for multi-agent

        state0 = agent.get_state()

        # assert (state0.numpy() == ego_states[-1]).all()

        ego_states = transform_poses(ego_states, state0)
        other_states = transform_poses(other_states, state0)

        dist = np.sqrt(other_states[...,0]**2 + other_states[...,1]**2)
        valid_masks = np.where(dist < self.perception_range, 1, 0)
        valid_lengths = valid_masks.sum(axis=1)

        _valid_masks_union = np.expand_dims(valid_masks, axis=2).repeat(other_states.shape[2], axis=2)
        valid_other_states = np.where(_valid_masks_union, other_states, self.invalid_value)
        valid_other_states = np.delete(valid_other_states, np.where(valid_lengths == 0)[0], axis=0)
        valid_masks = np.delete(valid_masks, np.where(valid_lengths == 0)[0], axis=0)

        ego_ht = self.historical_timestamps
        ego_states = np.concatenate([ego_states, ego_ht], axis=1)
        other_ht = np.expand_dims(self.historical_timestamps, axis=0).repeat(valid_other_states.shape[0], axis=0)
        valid_other_states = np.concatenate([valid_other_states, other_ht], axis=2)

        ### left align
        sorted_index = np.expand_dims(np.argsort(-valid_masks[:,-1], axis=0, kind='mergesort'), axis=1).repeat(valid_masks.shape[1],axis=1)
        valid_masks = np.take_along_axis(valid_masks, sorted_index, axis=0)
        valid_other_states = np.take_along_axis(valid_other_states, np.expand_dims(sorted_index, axis=2).repeat(valid_other_states.shape[2], axis=2), axis=0)
 
        ### normalize
        ego_states[:, :2] /= self.perception_range
        ego_states[:, 2] /= np.pi
        ego_states[:, 3] /= agent.max_velocity
        ego_states[:, -1] /= (self.horizon/self.decision_frequency)

        valid_other_states[:,:, :2] /= self.perception_range
        valid_other_states[:,:, 2] /= np.pi
        valid_other_states[:,:, 3] /= agent.max_velocity
        valid_other_states[:,:, -1] /= (self.horizon/self.decision_frequency)

        ### split state and character
        ego_character = agent.character
        ego_states = np.delete(ego_states, -2, axis=-1)
        other_characters = valid_other_states[...,[-2]]
        valid_other_states = np.delete(valid_other_states, -2, axis=-1)

        return rldev.Data(ego=ego_states, ego_mask=ego_masks, character=ego_character, obs=valid_other_states, obs_mask=valid_masks, obs_character=other_characters)




    def render(self, ax, data: List[rldev.Data], colors=None):
        raise NotImplementedError

    def render_vi(self, ax, data: rldev.Data, colors=None):
        print(rldev.prefix(self) + 'viz')
        ax.clear()

        if colors == None:
            colors = rldev.Data(
                ego=self.default_colors.ego,
                obs=[self.default_colors.obs] *len(data.obs),
                route=self.default_colors.route,
                lane=[self.default_colors.lane] *len(data.lane),
                bound=[self.default_colors.bound] *len(data.bound),
            )



        ### 0. perception circle
        circle_x, circle_y = 0, 0
        circle_r = self.perception_range /self.perception_range
        a_x = np.arange(0, 2*np.pi, 0.01)
        a = circle_x + circle_r *np.cos(a_x)
        b = circle_y + circle_r *np.sin(a_x)
        ax.plot(a, b, '-b')
        ax.plot(a,-b, '-b')

        ### 1.1 lane
        for lane, c in zip(data.lane, colors.lane):
            lane_x, lane_y = lane[:,0], lane[:,1]
            ax.plot(lane_x, lane_y, '-', color=c)

        ### 1.2 route
        route_x, route_y = data.route[:,0], data.route[:,1]
        ax.plot(route_x, route_y, '-', color=colors.route)

        ### 1.3 bound
        for bound, c in zip(data.bound, colors.bound):
            bound_x, bound_y = bound[:,0], bound[:,1]
            ax.plot(bound_x, bound_y, '-', color=c)

        ### 3. obs
        for obs, c in zip(data.obs, colors.obs):
            obs_x, obs_y = obs[:,0], obs[:,1]
            ax.plot(obs_x, obs_y, '-o', color=c)


        ### 4. ego
        ego_x, ego_y = data.ego[:,0], data.ego[:,1]
        ax.plot(ego_x, ego_y, '-o', color=colors.ego)


        # ax.savefig('{}.png'.format('results/tmp/'+str(self.time_step)))
        # ax.show()

        # import os
        # save_dir = os.path.join(self.config.path_pack.output_path, str(data.step_reset))
        # cu.system.mkdir(save_dir)
        # print(rldev.prefix(self) + 'save fig')
        # ax.savefig( os.path.join(save_dir, '{}.png'.format(self.time_step)) )
        return





class PerceptionVectorizedRoute(object):
    dim_state = 4
    num_points = 20

    def __init__(self, config, sampling_resolution=2.0):
        self.perception_range = config.perception_range
        self.sampling_resolution = sampling_resolution


    def run_step(self, vehicle: Vehicle):
        state0 = vehicle.get_state()
        global_path = vehicle.global_path
        global_path.step_coverage(vehicle.get_transform())
        s = global_path.max_coverage
        i = np.clip(int(self.sampling_resolution / global_path.sampling_resolution), 1, None)
        d = self.num_points
        x, y, theta = global_path.x[s::i][:d], global_path.y[s::i][:d], global_path.theta[s::i][:d]
        x, y, theta = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(theta, dtype=np.float32)
        spacestamps = np.arange(0, self.num_points, dtype=np.float32)/ self.num_points
        route = np.stack([x, y, theta, spacestamps[:len(x)]], axis=1)
        route = np.concatenate([
            route,
            np.full([self.num_points-route.shape[0], route.shape[1]], np.inf, dtype=np.float32),
        ], axis=0)
        route_mask = np.where(route < np.inf, 1,0).all(axis=-1)
        route = transform_poses(route, state0)

        ### normalize
        route[:,:2] /= self.perception_range
        route[:,2] /= np.pi

        return rldev.Data(route=route, route_mask=route_mask)





class PerceptionVectorizedMap(object):
    invalid_value = np.inf
    dim_state = 4

    def __init__(self, config, topology_map: TopologyMap):
        self.config = config
        self.topology_map = topology_map
        self.perception_range = config.perception_range
        return



    def run_step(self, step_reset, time_step, agent: EndToEndVehicle):
        self.step_reset, self.time_step = step_reset, time_step
        state0 = agent.get_state()

        lane, lane_mask = self.topology_map.crop_line(state0, self.perception_range, line_type='center')
        bound, bound_mask = self.topology_map.crop_line(state0, self.perception_range, line_type='side')

        bound_flag = self.check_in_bound(agent, bound)

        ### normalize
        lane[:,:,:3] /= self.perception_range
        lane[:,:,3] /= 50
        bound[:,:,:3] /= self.perception_range
        bound[:,:,3] /= 50
        return rldev.Data(lane=lane, lane_mask=lane_mask, bound=bound, bound_mask=bound_mask, bound_flag=bound_flag)



    def check_in_bound(self, agent: EndToEndVehicle, bound):
        #检测严格程度[0.1, 1]
        scale = 0.6
        #设定碰撞检测范围就是10m
        range = 30.0
        state0 = agent.get_state()
        bounds = bound[..., :2]

        #取距离小于10m的bound
        dist = np.sqrt(bounds[...,0]**2 + bounds[...,1]**2)

        ### ! warning
        if np.where(dist <= np.hypot(agent.bbx_x, agent.bbx_y) *scale +0.1, 1, 0).sum() == 0:
            return np.array(False)

        valid_masks = np.where(dist < range, 1, 0)
        valid_lengths = valid_masks.sum(-1)
        max_valid_lengths = max(valid_lengths)

        sorted_index = np.expand_dims(np.argsort(-valid_masks, axis=1, kind='mergesort'), axis=2).repeat(bounds.shape[2], axis=2)
        valid_masks = np.expand_dims(valid_masks, axis=2).repeat(bounds.shape[2], axis=2)
        valid_bounds = np.where(valid_masks, bounds, np.inf)
        valid_bounds = np.take_along_axis(valid_bounds, sorted_index, axis=1)[:,:max_valid_lengths]
        valid_bounds = np.delete(valid_bounds, np.where(valid_lengths == 0)[0], axis=0).copy()
        if(valid_bounds.size == 0):
            return np.array(False)

        valid_bounds_lines, valid_masks_lines = self.trans_points_to_lines(valid_bounds)
        agent_profile_lines = np.empty((4, 4), dtype=np.float32)

        theta = 0
        
        # cu.basic.pi2pi
        x, y = agent.bbx_x * scale, agent.bbx_y * scale
        ##车长两条线
        agent_profile_lines[0,0], agent_profile_lines[0,1]= -np.cos(theta)*x - np.sin(theta)*y, -np.sin(theta)*x + np.cos(theta)*y
        agent_profile_lines[0,2], agent_profile_lines[0,3]= np.cos(theta)*x - np.sin(theta)*y, np.sin(theta)*x + np.cos(theta)*y
        agent_profile_lines[1,0], agent_profile_lines[1,1]= -np.cos(theta)*x + np.sin(theta)*y, -np.sin(theta)*x - np.cos(theta)*y 
        agent_profile_lines[1,2], agent_profile_lines[1,3]= np.cos(theta)*x + np.sin(theta)*y, np.sin(theta)*x - np.cos(theta)*y
        #车宽两条线
        agent_profile_lines[2,0], agent_profile_lines[2,1]= agent_profile_lines[0,0], agent_profile_lines[0,1]
        agent_profile_lines[2,2], agent_profile_lines[2,3]= agent_profile_lines[1,0], agent_profile_lines[1,1]
        agent_profile_lines[3,0], agent_profile_lines[3,1]= agent_profile_lines[0,2], agent_profile_lines[0,3] 
        agent_profile_lines[3,2], agent_profile_lines[3,3]= agent_profile_lines[1,2], agent_profile_lines[1,3]

        cross_point_check_data_A = np.expand_dims(agent_profile_lines, axis = 1).repeat(valid_bounds_lines.shape[1], axis=1)
        cross_point_check_data_A = np.expand_dims(cross_point_check_data_A, axis = 1).repeat(valid_bounds_lines.shape[0], axis=1)
        cross_point_check_data_B = np.expand_dims(valid_bounds_lines, axis = 0).repeat(cross_point_check_data_A.shape[0], axis=0)
        cross_point_check_data = np.concatenate([cross_point_check_data_A, cross_point_check_data_B], axis = -1)
        valid_masks_lines = np.expand_dims(valid_masks_lines, axis = 0).repeat(cross_point_check_data.shape[0], axis = 0)
        valid_masks_lines = np.expand_dims(valid_masks_lines, axis = -1).repeat(2, axis = -1)

        check_point_res = self.cross_point_check(cross_point_check_data, valid_masks_lines)
        check_point_res = check_point_res.astype(np.int).sum().astype(np.bool)
        return check_point_res




    def trans_points_to_lines(self, points: np.array):
        #vec (a,b,2) -> lines (a,b-1,4)
        lines = np.empty((points.shape[0], points.shape[1] - 1, 4), dtype = np.float32)
        lines[:,:,:2]  = points[:,:-1,:2]
        lines[:,:,2:4] = points[:,1:,:2]
        valid_masks = np.where(lines[:,:,3] == np.inf, 0, 1)
        return lines, valid_masks


    def cross_point_check(self, cross_point_check_data: np.array, cross_point_check_masks: np.array) :
        # 参考:https://segmentfault.com/a/1190000004457595?f=tt ,将每条ployline的seg的四个线段和其他polyline做相交检测
        # cross_point_check_data.shape 是 [...,8] 最后一维是两条线段
        # cross_point_check_masks.shape 是 [...,2]
        cross_point_check_data = np.where(cross_point_check_data == np.inf, np.nan, cross_point_check_data)
        vec_AC =  np.where(cross_point_check_masks,cross_point_check_data[...,4:6] - cross_point_check_data[...,0:2], np.nan)
        vec_AD =  np.where(cross_point_check_masks,cross_point_check_data[...,6:8] - cross_point_check_data[...,0:2], np.nan)
        vec_BC =  np.where(cross_point_check_masks,cross_point_check_data[...,4:6] - cross_point_check_data[...,2:4], np.nan)
        vec_BD =  np.where(cross_point_check_masks,cross_point_check_data[...,6:8] - cross_point_check_data[...,2:4], np.nan)
        vec_CA, vec_CB, vec_DA, vec_DB = vec_AC, vec_BC, vec_AD, vec_BD
        ZERO = 1e-2
        vec_product_AC_AD = np.where(cross_point_check_masks[...,0],(vec_AC[...,0]*vec_AD[...,1] - vec_AD[...,0]*vec_AC[...,1]), np.inf)
        vec_product_BC_BD = np.where(cross_point_check_masks[...,0],(vec_BC[...,0]*vec_BD[...,1] - vec_BD[...,0]*vec_BC[...,1]), np.inf)
        vec_product_CA_CB = np.where(cross_point_check_masks[...,0],(vec_CA[...,0]*vec_CB[...,1] - vec_CB[...,0]*vec_CA[...,1]), np.inf)
        vec_product_DA_DB = np.where(cross_point_check_masks[...,0],(vec_DA[...,0]*vec_DB[...,1] - vec_DB[...,0]*vec_DA[...,1]), np.inf)
        bool_res_part_1 = np.array(np.where((vec_product_AC_AD*vec_product_BC_BD)<=ZERO, 1, 0), dtype=np.bool)
        bool_res_part_2 = np.array(np.where((vec_product_CA_CB*vec_product_DA_DB)<=ZERO, 1, 0), dtype=np.bool)
        cross_point_check_res = np.logical_and(bool_res_part_1, bool_res_part_2)
        # cross_point_check_res.shape 是 [...] 最后一维是两条线段,相当于少了一维,储存的是是否相交
        return cross_point_check_res


