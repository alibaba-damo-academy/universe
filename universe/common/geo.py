import rldev

import numpy as np




class Vector(object):
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __str__(self):
        return str(type(self))[8:-2] + f'(x={self.x}, y={self.y})'

    def __repr__(self):
        return str(self)


    def distance(self, location):
        return np.hypot(self.x - location.x, self.y - location.y)



    def render(self, ax, color=None, linewidth=1):
        ax.plot(self.x, self.y, 'or', linewidth=linewidth)
        return



class Transform(object):
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta


    def __str__(self):
        return str(type(self))[8:-2] + f'(x={self.x}, y={self.y}, theta={self.theta})'

    def __repr__(self):
        return str(self)



    def distance(self, transform):
        return np.hypot(self.x - transform.x, self.y - transform.y)



    def render(self, ax, color=None, length=0.01, width=0.02, linewidth=0.1, zorder=3):
        dx = length*np.cos(self.theta)
        dy = length*np.sin(self.theta)

        fc = ec = color
        ax.arrow(self.x, self.y, dx, dy, fc=fc, ec=ec, linewidth=linewidth, head_width=width, head_length=width, zorder=zorder)
        return




class HashableTransform(object):
    def __init__(self, transform: Transform, resolution=0.02):
        self.transform = transform
        self.resolution = resolution

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return self.transform.distance(other.transform) < self.resolution




def error_transform(current_transform: Transform, target_transform: Transform):    
    xr, yr, thetar = target_transform.x, target_transform.y, target_transform.theta
    theta_e = rldev.pi2pi_numpy(current_transform.theta - thetar)

    d = (current_transform.x - xr, current_transform.y - yr)
    t = (np.cos(thetar), np.sin(thetar))

    longitudinal_e, lateral_e = _cal_long_lat_error(d, t)
    return longitudinal_e, lateral_e, theta_e

def _cal_long_lat_error(d, t):
    '''
        Args:
            d, t: array-like
    '''
    dx, dy = d[0], d[1]
    tx, ty = t[0], t[1]
    longitudinal_e = dx*tx + dy*ty
    lateral_e = dx*ty - dy*tx
    return longitudinal_e, lateral_e










def error_state(current_state, target_state):
    xr, yr, thetar = target_state.x, target_state.y, target_state.theta
    theta_e = rldev.pi2pi(current_state.theta - thetar)

    d = (current_state.x - xr, current_state.y - yr)
    t = (np.cos(thetar), np.sin(thetar))

    longitudinal_e, lateral_e = _cal_long_lat_error(d, t)
    return longitudinal_e, lateral_e, theta_e







class State(object):
    def __init__(self, **kwargs):
        self.x = kwargs.get('x', 0.0)
        self.y = kwargs.get('y', 0.0)

        self.theta = rldev.pi2pi_numpy(kwargs.get('theta', 0.0))

        self.k = kwargs.get('k', 0.0)

        self.s = kwargs.get('s', 0.0)
        self.v = kwargs.get('v', 0.0)
        self.a = kwargs.get('a', 0.0)
        self.t = kwargs.get('t', 0.0)

        self.velocity = kwargs.get('velocity', np.zeros((3,1))).astype(np.float64)
        self.acceleration = kwargs.get('acceleration', np.zeros((3,1))).astype(np.float64)
        

    def __str__(self):
        obj = 'State(x={}, y={}, theta={}, v={})'.format(self.x, self.y, self.theta, self.v)
        return obj

    def __repr__(self):
        return str(self)


    def distance_xy(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        return np.sqrt(dx**2 + dy**2)

    def distance_xytheta(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        dtheta = self.theta - state.theta
        return np.sqrt(dx**2 + dy**2 + dtheta**2)

    def delta_theta(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        return np.arctan2(dy, dx)


    def find_nearest_state_in(self, states):
        min_dist = float('inf')
        min_state = None
        for state in states:
            d = self.distance(state)
            if d < min_dist:
                min_dist = d
                min_state = state
        return min_state

    def find_nearest_waypoint_in(self, waypoints):
        mind = float('inf')
        nearest_wp = None
        for waypoint in waypoints:
            d = self.distance(waypoint)
            if d < mind:
                mind = d
                nearest_wp = waypoint
        return nearest_wp, mind


    def world2local(self, state0):
        '''
            2-dimension
            state0: world coordinate
        '''

        x_world, y_world, theta_world = self.x, self.y, self.theta
        x0, y0, theta0 = state0.x, state0.y, state0.theta

        x_local = (x_world-x0)*np.cos(theta0) + (y_world-y0)*np.sin(theta0)
        y_local =-(x_world-x0)*np.sin(theta0) + (y_world-y0)*np.cos(theta0)
        delta_theta = rldev.pi2pi(theta_world - theta0)

        local_state = State(
            x=x_local, y=y_local, theta=delta_theta,
            k=self.k, s=self.s, v=self.v, a=self.a, t=self.t,
        )
        return local_state


    def local2world(self, state0):

        x_local, y_local, theta_local = self.x, self.y, self.theta
        x0, y0, theta0 = state0.x, state0.y, state0.theta

        x_world = x0 + x_local*np.cos(theta0) - y_local*np.sin(theta0)
        y_world = y0 + x_local*np.sin(theta0) + y_local*np.cos(theta0)
        theta_world = rldev.pi2pi(theta_local + theta0)

        world_state = State(
            x=x_world, y=y_world, theta=theta_world,
            k=self.k, s=self.s, v=self.v, a=self.a, t=self.t
        )
        return world_state


    def numpy(self):
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float32)






def transform2state(transform: Transform, **kwargs):
    x, y, theta = transform.x, transform.y, transform.theta
    kwargs.update({'x':x, 'y':y, 'theta':theta})
    return State(**kwargs)


def state2transform(state: State):
    return Transform(state.x, state.y, state.theta)

