import rldev

import numpy as np
import random

from .. import ScenarioRandomization as uni_ScenarioRandomization


class ScenarioRandomization(uni_ScenarioRandomization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.characters = self.get_characters()

    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        return characters.astype(np.float32)

    def __getitem__(self, vi):
        return super().__getitem__(vi) + rldev.BaseData(character=self.characters[vi])

