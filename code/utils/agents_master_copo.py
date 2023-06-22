import rllib

import numpy as np
import copy

import universe
from universe.common import Vehicle
from universe.common import VehicleControl
from universe.common import EndToEndVehicle
from universe.common import IdmVehicle, RuleVehicle



from .agents_master import EndToEndVehicleWithCharacter, AgentListMaster




class NeuralVehicleTuneSVO(EndToEndVehicleWithCharacter):
    dim_action = 1
    
    def set_svo(self, reference):
        character = np.clip(reference[0], -1,1)
        self.character = ((character + 1) *0.5).astype(np.float32)
        return







class AgentListMasterTuneSVO(AgentListMaster):
    """
        Only for multi agent without vehicles_rule.
    """
    
    def __init__(self, config: rllib.basic.YamlConfig, topology_map, **kwargs):
        super().__init__(config, topology_map, **kwargs)

        config_neural_policy = config.config_neural_policy
        config_neural_policy.set('dim_state', config.dim_state)
        config_neural_policy.set('dim_action', 2)
        self.buffer_cls = config_neural_policy.buffer
        self.device = config_neural_policy.device
        self.neural_policy = rllib.sac.Actor(config_neural_policy).to(self.device)
        self.neural_policy.load_model()


    def observe(self, step_reset, time_step):
        state = super().observe(step_reset, time_step)
        self.state = state
        return state


    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """

        assert len(references) == len(self.vehicles_neural)

        if self.config.scenario_name == 'bottleneck':
            svos = np.random.normal(0.8, 0.05, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'intersection_v2':
            svos = np.random.normal(0.9, 0.1, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'merge':
            svos = np.random.normal(0.5, 0.05, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'roundabout':
            svos = np.random.normal(0.8, 0.05, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        else:
            raise NotImplementedError(f'unkonwn {self.config.scenario_name}')

        state = [s.to_tensor().unsqueeze(0) for s in self.state]
        states = rllib.buffer.stack_data(state)
        self.buffer_cls.pad_state(None, states)
        states = states.cat(dim=0)
        actions, _, _ = self.neural_policy.sample(states.to(self.device))
        actions = actions.detach().cpu().numpy()

        vehicles = self.vehicles_neural
        [vehicle.set_svo(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, actions)]
        targets = targets_neural

        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return








class AgentListMasterTuneSVOAdv(AgentListMaster):
    """
        Only for multi agent without vehicles_rule.
    """
    
    def __init__(self, config: rllib.basic.YamlConfig, topology_map, **kwargs):
        super().__init__(config, topology_map, **kwargs)

        config_neural_policy = config.config_neural_policy
        config_neural_policy.set('dim_state', config.dim_state)
        config_neural_policy.set('dim_action', 2)
        self.buffer_cls = config_neural_policy.buffer
        self.device = config_neural_policy.device
        self.neural_policy = rllib.sac.Actor(config_neural_policy).to(self.device)
        self.neural_policy.load_model()


    def observe(self, step_reset, time_step):
        state = super().observe(step_reset, time_step)
        self.state = state
        return state


    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """

        assert len(references) == len(self.vehicles_neural)

        if self.config.scenario_name == 'bottleneck':
            svos = np.random.normal(0.0, 0.1, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'intersection_v2':
            svos = np.random.normal(0.0, 0.05, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'merge':
            svos = np.random.normal(1.0, 0.05, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'roundabout':
            svos = np.random.normal(1.0, 0.05, size=references.shape)
            references = (svos *2 -1).astype(np.float32)
        else:
            raise NotImplementedError(f'unkonwn {self.config.scenario_name}')

        state = [s.to_tensor().unsqueeze(0) for s in self.state]
        states = rllib.buffer.stack_data(state)
        self.buffer_cls.pad_state(None, states)
        states = states.cat(dim=0)
        actions, _, _ = self.neural_policy.sample(states.to(self.device))
        actions = actions.detach().cpu().numpy()

        vehicles = self.vehicles_neural
        [vehicle.set_svo(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, actions)]
        targets = targets_neural

        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return


