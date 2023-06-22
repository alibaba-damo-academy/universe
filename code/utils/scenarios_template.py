import rllib
import universe

import numpy as np
import random



class ScenarioRandomization(universe.ScenarioRandomization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.characters = self.get_characters()

    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)

    def __getitem__(self, vi):
        return super().__getitem__(vi) + rllib.basic.BaseData(character=self.characters[vi])



class ScenarioRandomization_share_character(ScenarioRandomization):
    def get_characters(self):
        character = random.uniform(0, 1)
        characters = [character] * self.num_vehicles
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters




class ScenarioRandomizationNegtiveSVOOld(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(-1,1, size=self.num_vehicles)
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)






class ScenarioRandomizationNegtiveSVO(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(-1,1, size=self.num_vehicles)
        # characters = np.clip(characters, -1, 0)
        characters[0] = -4.0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)



class ScenarioRandomizationAllNegtiveSVO(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(-1,0, size=self.num_vehicles)
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)






class ScenarioRandomizationNegtiveSVO_share_character(ScenarioRandomization):
    def get_characters(self):
        character = random.uniform(-1, 1)
        characters = [character] * self.num_vehicles
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters










class ScenarioRandomizationWithoutMismatch(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        characters[0] = 0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)





