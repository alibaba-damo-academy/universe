import rllib
import universe

import copy
import torch
import numpy as np

from universe.common import check_collision



class EnvMultiAgentAdv(universe.EnvInteractiveMultiAgent):
    def step(self, action):
        self.time_step += 1

        # import pdb; pdb.set_trace()
        # print('-----------------len: ', len(action), len(self.agents_master.vehicles_neural))

        # adv_character = (action_adv.item() + 1) *0.5
        # print('adv_character: ', adv_character)

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
            e = rllib.basic.Data(vi=s.vi,
                state=s.to_tensor().unsqueeze(0),
                action=torch.from_numpy(a).unsqueeze(0),
                # action_adv=torch.from_numpy(ad).unsqueeze(0),
                next_state=ns.to_tensor().unsqueeze(0),
                reward=torch.tensor([r], dtype=torch.float32),
                done=torch.tensor([d], dtype=torch.float32),
            )
            experience.append(e)

        ### metric end
        if done:
            self.metric.on_episode_end()
            self.recorder.save()
        return experience[0], experience[1:], done, episode_info.to_dict()




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


        finish = [ord|ort|wl|rb for (ord, ort, wl, rb) in zip(off_road, off_route, wrong_lane, reach_boundary)]
        finish[0] = False
        done = collision[0] or off_road[0] or off_route[0] or wrong_lane[0] or reach_boundary[0] or timeout


        episode_info = rllib.basic.BaseData(done=done, timeout=timeout,
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




class EnvMultiAgentAdvTarget(EnvMultiAgentAdv):
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
        finish[0] = False
        done = collision[0] or off_road[0] or off_route[0] or wrong_lane[0] or reach_boundary[0] or timeout


        episode_info = rllib.basic.BaseData(done=done, timeout=timeout,
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








from universe import Scenario
from universe import AgentsMaster
from universe import Metric


class MetricAdv(Metric):
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
            metrics_final[vi] = rllib.basic.Data(
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

            if vi == 0:
                self.writer.add_scalar(f'{self.env_name}/{vi}_steps', mf_vi.steps, self.step_reset)
                self.writer.add_scalar(f'{self.env_name}/{vi}_reward', mf_vi.reward, self.step_reset)
                self.writer.add_scalar(f'{self.env_name}/{vi}_speed', mf_vi.speed, self.step_reset)
                self.writer.add_scalar(f'{self.env_name}/{vi}_collision', mf_vi.collision, self.step_reset)
                self.writer.add_scalar(f'{self.env_name}/{vi}_off_road', mf_vi.off_road, self.step_reset)
                self.writer.add_scalar(f'{self.env_name}/{vi}_off_route', mf_vi.off_route, self.step_reset)
                self.writer.add_scalar(f'{self.env_name}/{vi}_wrong_lane', mf_vi.wrong_lane, self.step_reset)
                self.writer.add_scalar(f'{self.env_name}/{vi}_success', success, self.step_reset)
        
        # if not self.sa:
        if True:
            self.writer.add_scalar(f'{self.env_name}/union_speed', np.average(metrics['speed']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_collision', np.average(metrics['collision']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_off_road', np.average(metrics['off_road']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_off_route', np.average(metrics['off_route']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_wrong_lane', np.average(metrics['wrong_lane']), self.step_reset)
            self.writer.add_scalar(f'{self.env_name}/union_success', np.average(metrics['success']), self.step_reset)
        return




