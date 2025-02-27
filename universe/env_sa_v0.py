import rldev

import numpy as np
import torch
from enum import Enum
import copy
import os

from .common.actor import check_collision
from .dataset import Dataset
from .reward_func import RewardFunc
from .metric import Metric
from .recorder import Recorder, PseudoRecorder



class AgentMode(Enum):
    replay = 1
    interactive = 2



class SingleAgentBoxSpace(object):
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low,self.high, size=self.shape).astype(self.dtype)




class EnvInteractiveSingleAgent(object):
    sa = True

    def __init__(self, config: rldev.YamlConfig, writer: rldev.Writer, env_index=0):
        rldev.setup_seed(config.seed)
        self.config, self.path_pack = config, config.path_pack
        self.writer = writer
        self.env_index = env_index
        self.env_name = 'env' + str(env_index) + f'_{config.scenario_name}'
        self.output_dir = os.path.join(self.path_pack.output_path, self.env_name)
        rldev.mkdir(self.output_dir)

        self.mode: AgentMode = config.mode
        if self.mode == AgentMode.replay:
            self.world_tick = self.world_tick_replay
        elif self.mode == AgentMode.interactive:
            self.world_tick = self.world_tick_interactive
        else:
            raise NotImplementedError
        dataset_cls = config.get('dataset_cls', Dataset)
        self.datasets = [dataset_cls(config, env_index, case_id) for case_id in config.case_ids]
        self.num_cases = len(self.datasets)
        print(rldev.prefix(self) + 'num_cases: ', self.num_cases, '\n')

        reward_func_cls = config.get('reward_func', RewardFunc)
        metric_cls = config.get('metric_cls', Metric)
        recoder_cls = config.get('recorder_cls', PseudoRecorder)
        self.reward_func = reward_func_cls()
        self.metric = metric_cls(writer, self.env_name, self.sa)
        self.recorder = recoder_cls(config, self.env_name, self.sa, self.output_dir)

        self.dim_state = self.datasets[0].agents_master.dim_state
        self.dim_action = self.datasets[0].agents_master.dim_action

        if self.sa:
            self.slice = lambda: 0
        else:
            self.slice = lambda: slice(0, self.num_agents)

        self.step_reset = config.get('step_reset', 0) -1
        return
    

    def reset(self):
        self.step_reset += 1
        self.time_step = 0

        self.dataset = self.datasets[self.step_reset % self.num_cases]
        self.num_steps = len(self.dataset)

        self.scenario = self.dataset.scenario
        self.agents_master = self.dataset.agents_master
        self.agents_master.destroy()

        self.scenario.reset(self.step_reset, sa=self.sa)
        self.num_vehicles = self.scenario.num_vehicles
        self.num_vehicles_max = self.scenario.num_vehicles_max
        self.num_agents = self.scenario.num_agents
        self.agents_master.reset(self.num_steps)

        self.scenario.register_agents(self.agents_master)

        self.state = self.agents_master.observe(self.step_reset, self.time_step)

        self.metric.on_episode_start(self.step_reset, self.scenario, self.agents_master, self.num_steps, self.num_agents)
        self.recorder.record_scenario(self.step_reset, self.scenario, self.agents_master)
        return self.state[self.slice()]



    def step(self, action):
        self.time_step += 1

        episode_info = self.check_episode()
        reward = self.reward_func.run_step(self.state, action, self.agents_master, episode_info)
        done = episode_info.done

        ### metric step
        self.metric.on_episode_step(self.time_step, reward, episode_info)
        self.recorder.record_vehicles(self.time_step, self.agents_master, episode_info)

        ### step
        self.world_tick(action)

        ### reset done agents, only for multi-agent
        self.reset_done_agents(episode_info)

        ### next_state
        next_state = self.agents_master.observe(self.step_reset, self.time_step)
        state = self.state
        self.state = copy.copy(next_state)

        ### pad next_state, only for multi-agent
        next_state_vis = [s.vi for s in next_state]
        for s in state:
            if s.vi not in next_state_vis:
                next_state.append(s)
        next_state = sorted(next_state, key=lambda x: x.vi)

        ### experience
        experience = []
        for s, a, ns, r, d in zip(state, action, next_state, reward, episode_info.finish):
            e = rldev.Data(vi=s.vi,
                state=s.to_tensor().unsqueeze(0),
                action=torch.from_numpy(a).unsqueeze(0),
                next_state=ns.to_tensor().unsqueeze(0),
                reward=torch.tensor([r], dtype=torch.float32),
                done=torch.tensor([d], dtype=torch.float32),
            )
            experience.append(e)

        ### metric end
        if done:
            self.metric.on_episode_end()
            self.recorder.save()
        return experience[self.slice()], done, episode_info.to_dict()

    def render(self):
        import matplotlib.pyplot as plt
        if not hasattr(self, 'fig'):
            print(rldev.prefix(self) + 'init env figure')
            xr, yr = self.scenario.x_range, self.scenario.y_range
            if xr / yr > 3:
                dx, dy = 0.0, 10.0
                subplots_adjust = rldev.BaseData(
                    top=1, bottom=0.7,
                    left=0.05, right=0.95,
                    # hspace=0.2, wspace=0.2,
                )
                self.bbox_to_anchor = (0, -0.2)
                self.loc = 'upper left'
            else:
                dx, dy = 10.0, 0.0
                subplots_adjust = rldev.BaseData(
                    top=0.95, bottom=0.05,
                    left=0, right=0.55,
                    # hspace=0.2, wspace=0.2,
                )
                self.bbox_to_anchor = (1, 1)
                self.loc = 'upper left'
            print(rldev.prefix(self) + f'size: x: {xr*0.05}, y: {yr*0.05}, dx: {dx}, dy: {dy}')
            self.fig = plt.figure(figsize=(xr*0.05 +dx, yr*0.05 +dy), dpi=100)

            self.fig.canvas.set_window_title(self.env_name)
            self.ax = self.fig.add_subplot(1,1,1)
            self.ax.set_aspect('equal', adjustable='box')
            self.scenario.render(self.ax)
            if self.config.invert:
                self.ax.invert_xaxis()
                plt.pause(0.00001)
            # self.fig.tight_layout()  ### This should be called after all axes have been added
            self.fig.subplots_adjust(**subplots_adjust.to_dict())
            self.foreground = []

        self.ax.set_title(f'step_reset: {self.step_reset}, time_step: {self.time_step}')

        # import pdb; pdb.set_trace()
        legend = self.ax.get_legend()
        if legend != None:
            legend.remove()
        [i.remove() for i in self.foreground]
        [i.remove() for i in self.ax.patches]
        self.foreground = []
        self.ax.patches = []

        lines, patches = self.agents_master.render(self.ax)
        self.foreground.extend(lines)

        ### global_path, optional
        # for vehicle in self.agents_master.vehicles_neural:
        #     vehicle.global_path.render(self.ax, length=1.0, width=0.05, linewidth=2.0)


        # if self.time_step == 0:
        #     if self.step_reset > 0:
        #         [plt.close(fig) for fig in self.figs]

        #     self.figs = []
        #     self.axes = []
        #     for i in range(self.num_agents):
        #         fig_i = plt.figure()
        #         ax_i = fig_i.add_subplot(1,1,1)
        #         ax_i.set_aspect('equal', adjustable='box')
        #         self.figs.append(fig_i)
        #         self.axes.append(ax_i)
        #         if self.config.invert:
        #             ax_i.invert_xaxis()
        #             plt.pause(0.00001)
        # for state, ax in zip(self.state, self.axes):
        #     self.agents_master.perception.render_vi(ax, state)
        
        self.ax.patches = sorted(self.ax.patches, key=lambda p: p.get_alpha())
        self.ax.legend(handles=patches, bbox_to_anchor=self.bbox_to_anchor, loc=self.loc)
        plt.pause(0.00001)
        if self.config.render_save:
            save_dir = os.path.join(self.output_dir, f'episode_{self.step_reset}')
            if self.time_step == 0:
                rldev.mkdir(save_dir)
            self.fig.savefig(os.path.join(save_dir, f'{self.time_step}.png'))
            self.fig.savefig(os.path.join(save_dir, f'{self.time_step}.pdf'))
        return


    @property
    def action_space(self):
        return SingleAgentBoxSpace(-1.0, 1.0, shape=(1, self.dim_action), dtype=np.float32)



    def check_episode(self):
        timeout = self.time_step >= self.num_steps-1

        off_road = [s.bound_flag.item() for s in self.state]

        off_route = []
        for agent in self.agents_master.vehicles_neural:
            _, lateral_e, _ = agent.global_path.error(agent.get_transform())
            off_route.append(abs(lateral_e) > 6)
        
        wrong_lane = []
        for agent in self.agents_master.vehicles_neural:
            wl = False
            if not self.scenario.at_junction(agent) and self.scenario.wrong_lane(agent):
                wl = True
            wrong_lane.append(wl)

        collision = check_collision([v.bounding_box for v in self.agents_master.vehicles_neural + self.agents_master.vehicles_rule])
        collision = collision[:len(self.agents_master.vehicles_neural)]

        reach_boundary = []
        for agent in self.agents_master.vehicles_neural:
            rb = False
            if not self.scenario.in_boundary(agent) or agent.reach_goal():
                rb = True
            reach_boundary.append(rb)


        finish = [c|ord|ort|wl|rb for (c, ord, ort, wl, rb) in zip(collision, off_road, off_route, wrong_lane, reach_boundary)]
        done = (np.array(finish) == True).all() or timeout


        episode_info = rldev.BaseData(done=done, timeout=timeout,
            finish=finish,
            collision=collision, off_road=off_road, off_route=off_route, wrong_lane=wrong_lane,
            reach_boundary=reach_boundary,
        )
        # print('\n\n\n')
        # print('step reset: ', self.step_reset)
        # print('finish: ', episode_info.finish)
        # print('timeout: ', timeout)
        # print('collision: ', episode_info.collision)
        # print('off_road: ', episode_info.off_road)
        # print('off_route: ', episode_info.off_route)
        # print('wrong_lane: ', episode_info.wrong_lane)
        # print('reach_boundary: ', episode_info.reach_boundary)
        return episode_info



    def world_tick_replay(self, action):
        self.dataset[self.time_step]
        pass

    def world_tick_interactive(self, action):
        self.agents_master.run_step(action)
        return


    def reset_done_agents(self, episode_info):
        finish_vehicles = []
        for vehicle in self.agents_master.vehicles_rule:
            rb = False
            if not self.scenario.in_boundary(vehicle) or vehicle.reach_goal():
                rb = True
            if rb:
                finish_vehicles.append(vehicle)
        for vehicle in finish_vehicles:
            self.agents_master.remove(vehicle)
        return








class EnvReplaySingleAgent(EnvInteractiveSingleAgent):
    def reset(self):
        self.step_reset += 1
        self.time_step = 0

        self.dataset = self.datasets[self.step_reset % self.num_cases]
        self.num_steps = len(self.dataset)

        self.scenario = self.dataset.scenario
        self.agents_master = self.dataset.agents_master
        # self.agents_master.destroy()

        # self.scenario.reset(self.step_reset, sa=self.sa)
        self.num_vehicles = self.scenario.num_vehicles
        self.num_vehicles_max = self.scenario.num_vehicles_max
        self.num_agents = self.scenario.num_agents
        # self.agents_master.reset(self.num_steps)

        self.set_data()

        # self.scenario.register_agents(self.agents_master)

        self.state = self.agents_master.observe(self.step_reset, self.time_step)

        self.metric.on_episode_start(self.step_reset, self.scenario, self.agents_master, self.num_steps, self.num_agents)
        self.recorder.record_scenario(self.step_reset, self.scenario, self.agents_master)
        return self.state[self.slice()]





    def set_data(self):
        self.data = self.dataset[self.time_step +1]

        vehicle_future_states = self.data.vehicle_future_states
        for vehicle in self.agents_master.vehicles_neural + self.agents_master.vehicles_rule:
            if vehicle.vi in vehicle_future_states.keys():
                vehicle.future_states = vehicle_future_states[vehicle.vi]
        return


    def check_episode(self):
        return self.data.episode_info



    def world_tick_replay(self, action):
        if self.data.episode_info.done:
            return
        
        self.set_data()
        vehicle_states = self.data.vehicle_states

        for vehicle in self.agents_master.vehicles_neural + self.agents_master.vehicles_rule:
            next_state = vehicle_states.get(vehicle.vi, None)
            if next_state != None:
                vehicle.set_state(next_state)
                vehicle.tick()
        return

    def world_tick_interactive(self, action):
        pass


