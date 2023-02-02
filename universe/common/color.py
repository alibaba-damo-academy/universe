
from enum import Enum
import numpy as np


class ColorLib(Enum):
    red = (255, 0, 0)
    orange = (255, 165, 0)
    yellow = (255, 255, 0)
    green = (0, 255, 0)
    cyan = (0, 255, 255)
    blue = (0, 0, 255)
    purple = (160, 32, 240)
    pink = (255, 192, 203)

    grey = (190, 190, 190)
    dim_grey = (105, 105, 105)


    butter_0 = (252, 233, 79)
    butter_1 = (237, 212, 0)
    butter_2 = (196, 160, 0)



    chocolate_0 = (233, 185, 110)
    chocolate_1 = (193, 125, 17)
    chocolate_2 = (143, 89, 2)

    chameleon_0 = (138, 226, 52)
    chameleon_1 = (115, 210, 22)
    chameleon_2 = (78, 154, 6)

    sky_blue_0 = (114, 159, 207)
    sky_blue_1 = (52, 101, 164)
    sky_blue_2 = (32, 74, 135)

    plum_0 = (173, 127, 168)
    plum_1 = (117, 80, 123)
    plum_2 = (92, 53, 102)

    scarlet_red_0 = (239, 41, 41)
    scarlet_red_1 = (204, 0, 0)
    scarlet_red_2 = (164, 0, 0)

    aluminium_0 = (238, 238, 236)
    aluminium_1 = (211, 215, 207)
    aluminium_2 = (186, 189, 182)
    aluminium_3 = (136, 138, 133)
    aluminium_4 = (85, 87, 83)
    aluminium_4_5 = (66, 62, 64)
    aluminium_5 = (46, 52, 54)


    dodger_blue = (30, 144, 255)
    deep_sky_blue = (0, 191, 255)


    white = (255, 255, 255)
    black = (0, 0, 0)


    @staticmethod
    def normal(color):
        return np.array(color.value) /255

