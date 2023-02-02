import rldev

import numpy as np

from .scenario import Scenario
from .agents_master import AgentsMaster



class Metric(object):
    def __init__(self, writer: rldev.Writer, env_name, sa):
        self.writer = writer
        self.env_name = env_name
        self.sa = sa
        return


    def init(self):
        return rldev.Data(
            steps=0,
            footprint=np.zeros((self.num_steps-1,), dtype=np.float32),
            reward=np.zeros((self.num_steps-1,), dtype=np.float32),
            speed=np.zeros((self.num_steps-1,), dtype=np.float32),
            collision=np.zeros((self.num_steps-1,), dtype=np.int),
            off_road=np.zeros((self.num_steps-1,), dtype=np.int),
            off_route=np.zeros((self.num_steps-1,), dtype=np.int),
            wrong_lane=np.zeros((self.num_steps-1,), dtype=np.int),
        )


    def on_episode_start(self, step_reset, scenario: Scenario, agents_master: AgentsMaster, num_steps, num_agents):
        self.step_reset = step_reset
        self.scenario = scenario
        self.agents_master = agents_master
        self.num_steps = num_steps
        self.num_agents = num_agents

        self.metrics = {}
        for vehicle in agents_master.vehicles_neural:
            self.metrics[vehicle.vi] = self.init()
        return

    def on_episode_step(self, time_step, reward, episode_info):
        for i, vehicle in enumerate(self.agents_master.vehicles_neural):
            vi = vehicle.vi
            if vi not in self.metrics.keys():
                self.metrics[vi] = self.init()
            self.metrics[vi].steps += 1
            self.metrics[vi].footprint[time_step-1] = 1
            self.metrics[vi].reward[time_step-1] = reward[i]
            self.metrics[vi].speed[time_step-1] = np.clip(vehicle.get_state().v /vehicle.max_velocity, 0,1)
            self.metrics[vi].collision[time_step-1] = episode_info.collision[i]
            self.metrics[vi].off_road[time_step-1] = episode_info.off_road[i]
            self.metrics[vi].off_route[time_step-1] = episode_info.off_route[i]
            self.metrics[vi].wrong_lane[time_step-1] = episode_info.wrong_lane[i]
        return

    def on_episode_end(self):
        remaining_agents = {agent.vi: agent for agent in self.agents_master.vehicles_neural}
        dtype = np.dtype([
            ('speed', np.float32, (1,)),
            ('collision', np.float32, (1,)),
            ('off_road', np.float32, (1,)),
            ('off_route', np.float32, (1,)),
            ('wrong_lane', np.float32, (1,)),
            ('success', np.float32, (1,)),
        ])
        metrics = np.array([], dtype=dtype)
        metrics_final = {}
        for vi, metric in self.metrics.items():
            if vi in remaining_agents.keys():
                status = self.scenario.finish_task(remaining_agents[vi])
            else:
                status = True
            metrics_final[vi] = rldev.Data(
                steps=metric.steps,
                footprint=metric.footprint,
                reward=metric.reward.sum(),
                speed=np.average(metric.speed),
                collision=metric.collision.any(),
                off_road=metric.off_road.any(),
                off_route=metric.off_route.any(),
                wrong_lane=metric.wrong_lane.any(),
                status=status,
            )
            mf_vi = metrics_final[vi]
            success = mf_vi.status *(1-mf_vi.collision) *(1-mf_vi.off_road) *(1-mf_vi.off_route) *(1-mf_vi.wrong_lane)
            metrics = np.append(metrics,
                np.array([(mf_vi.speed, mf_vi.collision, mf_vi.off_road, mf_vi.off_route, mf_vi.wrong_lane, success)], dtype=dtype)
            )
            # print(f'metrics vi {vi}: ', mf_vi, f',   success={success}')
            # import pdb; pdb.set_trace()

            # self.writer.add_scalar(f'{self.env_name}/steps_'+str(vi), mf_vi.steps, self.step_reset)
            # self.writer.add_scalar(f'{self.env_name}/reward_'+str(vi), mf_vi.reward, self.step_reset)
            # self.writer.add_scalar(f'{self.env_name}/speed_'+str(vi), mf_vi.speed, self.step_reset)
            # self.writer.add_scalar(f'{self.env_name}/collision_'+str(vi), mf_vi.collision, self.step_reset)
            # self.writer.add_scalar(f'{self.env_name}/off_road_'+str(vi), mf_vi.off_road, self.step_reset)
            # self.writer.add_scalar(f'{self.env_name}/off_route_'+str(vi), mf_vi.off_route, self.step_reset)
            # self.writer.add_scalar(f'{self.env_name}/wrong_lane_'+str(vi), mf_vi.wrong_lane, self.step_reset)
            # self.writer.add_scalar(f'{self.env_name}/success_'+str(vi), mf_vi.success, self.step_reset)
        
        # if not self.sa:
        if True:
            self.writer.add_scalar(f'{self.env_name}/union_speed', np.average(metrics['speed']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_collision', np.average(metrics['collision']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_off_road', np.average(metrics['off_road']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_off_route', np.average(metrics['off_route']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_wrong_lane', np.average(metrics['wrong_lane']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_success', np.average(metrics['success']), self.step_reset)
        return



