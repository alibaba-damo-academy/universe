import rldev

import numpy as np
import networkx as nx
import time

from .color import ColorLib
from .geo import Vector, State
from .global_path import GlobalPath


class TopologyMap(object):
    def __init__(self, centerline, sideline):
        """
            centerline, sideline: shape is (num_lines, num_points, num_features), left align

            example: >>> centerline[...,0]
                array([[ 10.047244,   8.092004,   5.754285,   3.797974,   2.400747,   1.775343,   1.75    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ -1.75    ,  -1.75    ,  -1.909573,  -2.417402,  -3.249022,  -4.392395,  -5.781641,  -7.309874, -10.026098,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ -5.25    ,  -5.250964,  -5.509147,  -6.198661,  -7.303393,  -8.669508, -10.820189,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 47.209763,  47.634853,  48.31167 ,  49.30091 ,  50.47067 ,  51.837143,  54.277905,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 43.7967  ,  44.264294,  45.232597,  46.72266 ,  48.867016,  51.39568 ,  54.277905,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 54.27791 ,  52.283344,  50.864864,  49.59009 ,  48.513767,  47.69643 ,  47.0525  ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [-18.793129, -16.86257 , -14.899687, -12.908382, -10.871655,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [-19.178913, -17.426746, -15.641137, -13.825632, -10.100035,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [110.      , 108.      , 106.      , 104.      , 102.      , 100.      ,  98.      ,  96.      ,  94.      ,  92.      ,  90.      ,  88.      ,  86.      ,  84.      ,  82.      ,  80.      ,  78.      ,  76.      ,  74.      ,  72.      ,  70.      ,  68.      ,  66.      ,  63.999996,  61.999996,  59.999996,  57.999996,  54.27791 ],
                    [110.      , 108.      , 106.      , 104.      , 102.      , 100.      ,  98.      ,  96.      ,  94.      ,  92.      ,  90.      ,  88.      ,  86.      ,  84.      ,  82.      ,  80.      ,  78.      ,  76.      ,  74.      ,  72.      ,  70.      ,  68.      ,  66.      ,  63.999996,  61.999996,  59.999996,  57.999996,  54.27791 ],
                    [ 54.277905,  56.277905,  58.277905,  60.277905,  62.277905,  64.27791 ,  66.27791 ,  68.27791 ,  70.27791 ,  72.27791 ,  74.27791 ,  76.27791 ,  78.27791 ,  80.27791 ,  82.27791 ,  84.27791 ,  86.27791 ,  88.27791 ,  90.27791 ,  92.27791 ,  94.27791 ,  96.27791 ,  98.27791 , 100.27791 , 102.27791 , 104.27791 , 106.27791 , 108.27791 ],
                    [ 54.277905,  56.277905,  58.277905,  60.277905,  62.277905,  64.27791 ,  66.27791 ,  68.27791 ,  70.27791 ,  72.27791 ,  74.27791 ,  76.27791 ,  78.27791 ,  80.27791 ,  82.27791 ,  84.27791 ,  86.27791 ,  88.27791 ,  90.27791 ,  92.27791 ,  94.27791 ,  96.27791 ,  98.27791 , 100.27791 , 102.27791 , 104.27791 , 106.27791 , 108.27791 ],
                    [-10.871655,  -8.835234,  -6.782441,  -4.717353,  -2.644074,  -0.566724,   1.487747,   3.560065,   5.62568 ,   7.680755,  10.849095,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [-10.100035,  -8.216515,  -6.31785 ,  -4.407815,  -2.490203,  -0.568824,   1.376155,   3.297817,   5.213262,   7.118934,  10.056937,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [-10.820189, -12.831341, -14.818178, -16.776936, -18.703903,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [-10.026098, -11.889651, -13.730676, -15.54568 , -17.331226, -19.083931,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 10.843021,   8.811576,   6.763432,   4.702472,   2.6326  ,   0.55774 ,  -1.518177,  -3.591217,  -5.657453,  -7.712966, -10.820189,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 10.047244,   8.164887,   6.267059,   4.357354,   2.439392,   0.516807,  -1.406757,  -3.327655,  -5.242248,  -7.146906, -10.026098,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 34.08578 ,  32.61534 ,  31.070187,  29.466154,  27.806286,  26.093721,  24.33171 ,  22.52359 ,  20.67279 ,  18.782814,  16.857244,  14.899731,  12.91398 ,  10.843021,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 31.614218,  30.221678,  28.789925,  27.303614,  25.765562,  24.178686,  22.54599 ,  20.87057 ,  19.155602,  17.404331,  15.620081,  13.806229,  10.047244,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 47.0525  ,  46.527184,  45.915833,  45.21958 ,  44.439724,  43.57771 ,  42.63514 ,  41.61377 ,  40.51548 ,  39.34234 ,  38.096508,  36.780304,  34.08578 ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 43.64818 ,  43.161053,  42.594147,  41.948513,  41.22535 ,  40.426003,  39.551956,  38.60484 ,  37.5864  ,  36.498543,  35.34328 ,  34.122765,  31.614218,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 33.981075,  35.437496,  36.83884 ,  38.17058 ,  39.43008 ,  40.614845,  41.722538,  42.750954,  43.69807 ,  44.562008,  45.341057,  46.033676,  46.6385  ,  47.209763,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 31.538921,  32.906597,  34.202934,  35.43488 ,  36.599995,  37.69598 ,  38.720665,  39.67202 ,  40.54816 ,  41.34736 ,  42.06803 ,  42.708748,  43.268246,  43.7967  ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ -5.25    ,  -5.25    ,  -5.25    ,  -5.25    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ -1.75    ,  -1.75    ,  -1.75    ,  -1.75    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [  1.75    ,   1.75    ,   1.75    ,   1.75    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [  5.25    ,   5.25    ,   5.25    ,   5.25    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 10.84302 ,   9.082572,   7.664848,   6.457124,   5.65864 ,   5.276662,   5.25    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [-10.871655,  -9.09983 ,  -7.674747,  -6.45651 ,  -5.655504,  -5.274981,  -5.25    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [  5.25    ,   5.25    ,   5.25    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [  1.75    ,   1.75    ,   1.75    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ -1.75    ,  -1.75    ,  -1.75    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ -5.25    ,  -5.25    ,  -5.25    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 10.849095,  12.85993 ,  14.846783,  16.805958,  18.73382 ,  20.626781,  22.48133 ,  24.294018,  26.061472,  27.780416,  29.447647,  31.060078,  33.981075,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 10.056937,  11.921585,  13.763994,  15.580738,  17.368444,  19.12379 ,  20.843513,  22.52442 ,  24.16338 ,  25.757359,  27.303383,  28.798588,  31.538921,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [  1.75    ,   1.75    ,   1.867888,   2.313691,   3.073008,   4.116317,   5.396147,   6.856528,  10.056937,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 47.209763,  47.62479 ,  47.948875,  48.181362,  48.321804,  48.36992 ,  48.33223 ,  48.206932,  47.992474,  47.689262,  47.0525  ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 43.796703,  44.18063 ,  44.480427,  44.69549 ,  44.825413,  44.86992 ,  44.834873,  44.71868 ,  44.519817,  44.238647,  43.64818 ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [ 54.27791 ,  52.27791 ,  50.598797,  49.05654 ,  47.65291 ,  46.428738,  45.367542,  44.536396,  43.64818 ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [  5.25    ,   5.250018,   5.483357,   6.151338,   7.238269,   8.598406,  10.849095,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf],
                    [-10.100035,  -7.920427,  -5.857324,  -3.982911,  -2.491257,  -1.791075,  -1.75    ,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf,        inf]], dtype=float32)
        """

        self.centerline = centerline
        self.centerline_mask = np.where(centerline < np.inf, True, False).all(axis=-1)
        self.centerline_length = self.centerline_mask.astype(np.int64).sum(axis=1)

        self.sideline = sideline
        self.sideline_mask = np.where(sideline < np.inf, True, False).all(axis=-1)
        self.sideline_length = self.sideline_mask.astype(np.int64).sum(axis=1)
        
        dx = np.diff(np.where(self.centerline_mask, self.centerline[...,0], 0.0))
        dy = np.diff(np.where(self.centerline_mask, self.centerline[...,1], 0.0))
        theta = np.arctan2(dy, dx)
        self.centerline_theta = np.where(self.centerline_mask, np.concatenate([theta, theta[:,[-1]]], axis=1), np.inf)
        for i, cl in enumerate(self.centerline_length):
            self.centerline_theta[i,cl-1] = self.centerline_theta[i,cl-2]

        self.graph = nx.DiGraph()

        t1 = time.time()
        self.build_graph()
        t2 = time.time()
        print(rldev.prefix(self) + 'build graph time: ', t2-t1)



    def build_graph(self):
        def add_node(i):
            segment = self.centerline[i]
            mask = self.centerline_mask[i]
            length = self.centerline_length[i]
            self.graph.add_node(i,
                segment=segment[mask],
                segment_center=segment[mask][:,:2].sum(axis=0) / length,
                segment_start=segment[mask][0,:2],
                segment_end=segment[mask][-1,:2],
            )
        def distance(p1, p2):
            dp = p1 - p2
            return np.hypot(dp[0], dp[1])


        for i in range(len(self.centerline)):
            add_node(i)
        
        for i in range(len(self.centerline)):
            node_i = rldev.Data(**self.graph.nodes[i])

            for j in range(len(self.centerline)):
                if i == j:
                    continue
                node_j = rldev.Data(**self.graph.nodes[j])

                if distance(node_i.segment_end, node_j.segment_start) < 0.05:
                    self.graph.add_edge(i, j)
                if distance(node_j.segment_end, node_i.segment_start) < 0.05:
                    self.graph.add_edge(j, i)
        return



    def route_planning(self, start: Vector, end: Vector):
        centerline_x, centerline_y = self.centerline[...,0], self.centerline[...,1]
        def find_closest_index(vec: Vector):
            dist = np.sqrt((centerline_x-vec.x)**2 + (centerline_y-vec.y)**2)
            return np.unravel_index(np.argmin(dist), dist.shape)
        
        index_start = find_closest_index(start)
        index_end = find_closest_index(end)

        indices = nx.astar_path(self.graph, index_start[0], index_end[0])
        segments = []
        for i in indices:
            segment = self.graph.nodes[i]['segment']
            if i == index_start[0]:
                segment = segment[index_start[1]:]
            if i == index_end[0]:
                segment = segment[:index_end[1]]
            
            if i != index_start[0]:
                segment = segment[1:]
            segments.append(segment)

        path = np.concatenate(segments, axis=0)
        return GlobalPath(path[:,0], path[:,1])




    def crop_line(self, state0: State, perception_range, line_type):
        if line_type == 'center':
            line = self.centerline
        elif line_type == 'side':
            line = self.sideline
        else:
            raise NotImplementedError
        
        center_x, center_y = state0.x, state0.y

        dist = np.sqrt((line[...,0]-center_x)**2 + (line[...,1]-center_y)**2)
        valid_masks = np.where(dist < perception_range, 1, 0)
        valid_lengths = valid_masks.sum(axis=1)
        max_valid_lengths = max(valid_lengths)

        valid_masks_union = np.expand_dims(valid_masks, axis=2).repeat(line.shape[2], axis=2)
        index_union = np.expand_dims(np.argsort(-valid_masks, axis=1, kind='mergesort'), axis=2).repeat(line.shape[2], axis=2)
        valid_topology_union = np.where(valid_masks_union, line, np.inf)
        valid_topology_union = np.take_along_axis(valid_topology_union, index_union, axis=1)[:,:max_valid_lengths]
        valid_topology_union = np.delete(valid_topology_union, np.where(valid_lengths == 0)[0], axis=0).copy()

        valid_masks = np.take_along_axis(valid_masks, index_union[:,:,0], axis=1)[:,:max_valid_lengths]
        valid_masks = np.delete(valid_masks, np.where(valid_lengths == 0)[0], axis=0).copy()

        valid_topology_union = transform_points(valid_topology_union, state0)

        ### concatenate
        spacestamps = np.expand_dims(np.arange(valid_topology_union.shape[1], dtype=np.float32), axis=(0,2)).repeat(valid_topology_union.shape[0], axis=0)
        valid_topology_union = np.concatenate([valid_topology_union, spacestamps], axis=2)
        valid_topology_union = np.where(
            np.expand_dims(valid_masks, axis=2).repeat(valid_topology_union.shape[2], axis=2),
            valid_topology_union,
            np.inf,
        )
        return valid_topology_union, valid_masks








    def render_graph(self, ax):
        pos = {key: rldev.Data(**self.graph.nodes[key]).segment_center for key in self.graph.nodes}

        edge_width = 1.0
        nx.draw_networkx_nodes(self.graph, pos, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, width=edge_width, arrowstyle='->', ax=ax)
        return

    def render(self, ax):
        for segment in self.centerline:
            ax.plot(segment[:,0], segment[:,1], '-', color=ColorLib.normal(ColorLib.grey), linewidth=1.0)

        for segment in self.sideline:
            ax.plot(segment[:,0], segment[:,1], '-', color=ColorLib.normal(ColorLib.dim_grey), linewidth=2.0)
        return









def transform_points(states: np.ndarray, state0: State):
    x, y = states[...,0], states[...,1]
    x0, y0, theta0 = state0.x, state0.y, state0.theta
    theta0 = theta0 + 1e-6 if abs(theta0) < 1e-7 else theta0

    ### https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
    # np.seterr(all='raise')

    x1, x2 = (x-x0)*np.cos(theta0), (y-y0)*np.sin(theta0)
    x_local = np.where(x1 == -np.inf, np.inf, x1) + np.where(x2 == -np.inf, np.inf, x2)
    y1, y2 = -(x-x0)*np.sin(theta0), (y-y0)*np.cos(theta0)
    y_local =np.where(y1 == -np.inf, np.inf, y1) + np.where(y2 == -np.inf, np.inf, y2)

    states_remaining = states[...,2:]
    axis = len(x_local.shape)
    states_local = np.concatenate([np.stack([x_local, y_local], axis=axis), states_remaining], axis=axis)
    return states_local



def transform_poses(states: np.ndarray, state0: State):
    x, y, theta = states[...,0], states[...,1], states[...,2]
    x0, y0, theta0 = state0.x, state0.y, state0.theta
    theta0 = theta0 + 1e-6 if abs(theta0) < 1e-7 else theta0

    theta_local = rldev.pi2pi(np.where(theta == np.inf, 0, theta) - theta0)
    theta_local = np.where(theta == np.inf, theta, theta_local)

    x1, x2 = (x-x0)*np.cos(theta0), (y-y0)*np.sin(theta0)
    x_local = np.where(x1 == -np.inf, np.inf, x1) + np.where(x2 == -np.inf, np.inf, x2)
    y1, y2 = -(x-x0)*np.sin(theta0), (y-y0)*np.cos(theta0)
    y_local =np.where(y1 == -np.inf, np.inf, y1) + np.where(y2 == -np.inf, np.inf, y2)

    states_remaining = states[...,3:]
    axis = len(x_local.shape)
    states_local = np.concatenate([np.stack([x_local, y_local, theta_local], axis=axis), states_remaining], axis=axis)
    return states_local



