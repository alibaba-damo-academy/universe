import rldev

import numpy as np
import copy



from .geo import Transform, error_transform


class HashableTransform(object):
    def __init__(self, transform: Transform, resolution=0.02):
        self.transform = transform
        self.resolution = resolution

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return self.transform.location.distance(other.transform.location) < self.resolution









def calc_curvature_with_yaw_diff(x, y, yaw):
    a, b = np.diff(x), np.diff(y)
    dists = np.hypot(a, b)
    d_yaw = rldev.pi2pi(np.diff(yaw))

    curvatures = d_yaw / dists
    curvatures = np.concatenate([curvatures, [0.0]])
    return curvatures, dists







class GlobalPath(object):
    def __init__(self, x, y):
        assert len(x) == len(y)
        assert len(x) > 1

        self.x, self.y = x, y

        dx, dy = np.diff(x), np.diff(y)
        theta = np.arctan2(dy, dx)
        self.theta = np.append(theta, theta[-1])

        self.transforms = []
        for (xi, yi, thetai) in zip(self.x, self.y, self.theta):
            t = Transform(xi, yi, thetai)
            self.transforms.append(t)

        self.curvatures, self.distances = calc_curvature_with_yaw_diff(self.x, self.y, self.theta)
        self.sampling_resolution = np.average(self.distances) if len(self) > 1 else 0.1

        self._max_coverage = 0
        return


    def __len__(self):
        return len(self.x)


    def render(self, ax, color=None, length=0.01, width=0.02, linewidth=0.1):
        start_x, start_y = self.x, self.y
        end_x = start_x + length*np.cos(self.theta)
        end_y = start_y + length*np.sin(self.theta)

        fc = ec = color
        for (x1, y1, x2, y2) in zip(start_x, start_y, end_x, end_y):
            if x1 == np.inf:
                continue
            ex, ey = 1e-4*np.cos(0.0), 1e-4*np.sin(0.0)
            ax.arrow(x1, y1, x2-x1 +ex, y2-y1 +ey, fc=fc, ec=ec, linewidth=linewidth, head_width=width, head_length=width)
        return




    @property
    def origin(self):
        return self.transforms[0]
    @property
    def destination(self):
        return self.transforms[-1]
    @property
    def max_coverage(self):
        return self._max_coverage
    
    def reached(self, preview_distance=0):
        preview_distance = max(0, preview_distance)
        preview_index = max(preview_distance // (self.sampling_resolution+1e-6) + 1, 0)
        return self._max_coverage >= len(self)-1 - preview_index

    


    def target_transform(self, current_transform):
        self.step_coverage(current_transform)
        index = min(len(self)-1, self._max_coverage+1)
        return self.transforms[index], self.curvatures[index]
    



    
    def remaining_transforms(self, current_transform):
        self.step_coverage(current_transform)
        return self.transforms[self._max_coverage:], sum(self.distances[self._max_coverage:])
    

    def step_coverage(self, current_transform):
        '''
            Args:
                current_transform: Transform
        '''
        index = self._max_coverage
        for index in range(self._max_coverage, len(self)):
            longitudinal_e, _, _ = error_transform(current_transform, self.transforms[min(len(self)-1, index+1)])
            if longitudinal_e < 0:
                break
        self._max_coverage = index


    def error(self, current_transform):
        self.step_coverage(current_transform)
        longitudinal_e, lateral_e, theta_e = error_transform(current_transform, self.transforms[self._max_coverage])
        return longitudinal_e, lateral_e, theta_e

