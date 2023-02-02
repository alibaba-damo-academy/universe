import rldev

import numpy as np
from typing import List
from shapely.geometry import Polygon

from .geo import Transform




class ActorBoundingBox(object):
    def __init__(self, transform: Transform, x, y):
        """
            x: length /2
            y: width /2
        """

        self.transform = transform
        self.x = x
        self.y = y
        
        self.vertices = ActorVertices.d2(self)
        return


    def intersects(self, bbx):
        p1 = Polygon(self.vertices)
        p2 = Polygon(bbx.vertices)
        return p1.intersects(p2)



    def render(self, ax, color=None, label=None, alpha=0.75):
        import matplotlib.patches as patches

        origin = ActorVertices.origin(self)
        r = patches.Rectangle(origin.point, origin.y_length, origin.x_length, linewidth=0, angle=np.rad2deg(origin.theta)-90, color=color, alpha=alpha, label=label, zorder=4)
        patch = ax.add_patch(r)
        return patch

    def render_polyline(self, ax, color=None, linewidth=1.0):
        vertices = ActorVertices.d2(self)
        vertices = np.vstack([vertices, vertices[[-1]], vertices[[0]]])
        # vertices, lines = ActorVertices.d2arrow(self)
        # vertices = np.vstack([vertices, vertices[[4]], vertices[[0]]])
        line = ax.plot(vertices[:,0], vertices[:,1], '-', color=color, linewidth=linewidth, zorder=5)[0]
        return line




class ActorVertices(object):
    @staticmethod
    def d2(actor_bbx, expand=rldev.BaseData(x=0.0, y=0.0)):
        t = actor_bbx.transform
        dx, dy = actor_bbx.x +expand.x, actor_bbx.y +expand.y
        center_x, center_y, theta = t.x, t.y, t.theta

        l, n = np.array([np.cos(theta), np.sin(theta)]), np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
        vertices = np.expand_dims(np.array([center_x, center_y]), axis=0).repeat(4, axis=0)

        vertices[0] +=  l*dx + n*dy
        vertices[1] += -l*dx + n*dy
        vertices[2] += -l*dx - n*dy
        vertices[3] +=  l*dx - n*dy
        return vertices
    
    @staticmethod
    def d2arrow(actor_bbx, expand=rldev.BaseData(x=0.0, y=0.0)):
        t = actor_bbx.transform
        dx, dy = actor_bbx.x +expand.x, actor_bbx.y +expand.y
        center_x, center_y, theta = t.x, t.y, t.theta

        l, n = np.array([np.cos(theta), np.sin(theta)]), np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
        vertices = np.expand_dims(np.array([center_x, center_y]), axis=0).repeat(7, axis=0)

        vertices[0] +=  l*dx + n*dy
        vertices[1] += -l*dx + n*dy
        vertices[2] += -l*dx - n*dy
        vertices[3] +=  l*dx - n*dy

        vertices[4] += l*dx
        vertices[5] += 0.3*l*dx + n*dy
        vertices[6] += 0.3*l*dx - n*dy

        lines = np.array([[0,1], [1,2], [2,3], [3,0],  [4,5], [5,6], [6,4]])
        return vertices, lines

    @staticmethod
    def origin(actor_bbx):
        t = actor_bbx.transform
        dx, dy = actor_bbx.x, actor_bbx.y
        center_x, center_y, theta = t.x, t.y, t.theta

        l, n = np.array([np.cos(theta), np.sin(theta)]), np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
        center = np.array([center_x, center_y])
        point = center -l*dx + n*dy
        return rldev.BaseData(point=point, theta=theta, x_length=dx*2, y_length=dy*2)





def check_collision(bbxs: List[ActorBoundingBox]):
    num_actors = len(bbxs)
    collisions = [False] * num_actors
    for i, bbx in enumerate(bbxs):
        if collisions[i] == True:
            continue
        for j, other_bbx in enumerate(bbxs):
            if i == j:
                continue
            if bbx.intersects(other_bbx):
                collisions[i] = True
                # collisions[j] = True
    return np.array(collisions)

