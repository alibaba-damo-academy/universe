import rllib

import os

from .scenarios_template import ScenarioRandomization
from universe.template.scenarios_intersection import ScenarioIntersection




############################################################################################
#### evaluate ##############################################################################
############################################################################################



class ScenarioIntersectionEvaluate(ScenarioIntersection):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return



class ScenarioIntersectionEvaluate_assign(ScenarioIntersectionEvaluate):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[:] = round(self.env_index *0.1111111, 7)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.num_vehicles, self.scenario_randomization.characters)
        return



class ScenarioIntersectionEvaluate_without_mismatch(ScenarioIntersectionEvaluate):  ### only for single-agent
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[0] = 0
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return




class ScenarioIntersectionEvaluate_without_mismatch__all_zero(ScenarioIntersectionEvaluate):  ### only for single-agent
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[:] = 0
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return




class ScenarioIntersectionEvaluateExplicitAdv(ScenarioIntersection):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters *= 2
        self.scenario_randomization.characters -= 1
        self.scenario_randomization.characters[0] = -4.0
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return




