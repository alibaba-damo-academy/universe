
import numpy as np
from typing import List


from .geo import Transform, error_transform
from .actor import ActorBoundingBox
from .vehicle import Vehicle


class RuleVehicle(Vehicle):
    pass





class StaticVehicle(RuleVehicle):
    def get_target(self, reference):
        return 0.0



class IdmVehicle(RuleVehicle):
    def __init__(self, config, vi, global_path, transform):
        super().__init__(config, vi, global_path, transform)

        self.leading_range = 50.0
        self.idm_scale_x = config.idm_scale_x
        self.idm_scale_y = config.idm_scale_y
        assert self.leading_range <= self.perception_range

        self.desired_speed = self.max_velocity *2
        self.time_gap = 1.0  ## 1.6
        self.min_gap = 2.0
        self.delta = 4.0
        self.acceleration = 1.0
        self.acceleration = self.max_acceleration  ### !warning
        self.comfortable_deceleration = 1.5  ## 1.7
        self.comfortable_deceleration = -self.min_acceleration  ### !warning
    

    def get_target(self, reference: List[Vehicle]):
        vehicles = reference

        '''get leading vehicle'''
        current_transform = self.get_transform()
        remaining_transforms, remaining_distance = self.global_path.remaining_transforms(current_transform)
        vehicle, transform, distance = get_leading_vehicle(self, vehicles, remaining_transforms, max_distance=self.leading_range, scale_x=self.idm_scale_x, scale_y=self.idm_scale_y)
        self.leading_vehicle = vehicle

        target_v = self.desired_speed
        if vehicle != None:
            target_v = self.intelligent_driver_model(vehicle, transform, distance)
        return target_v


    def intelligent_driver_model(self, leading_vehicle: Vehicle, leading_transform: Transform, leading_distance):
        distance_c2c = leading_distance
        length_two_half = leading_vehicle.bbx_x + self.bbx_x
        distance_b2b = distance_c2c - length_two_half - 0.3   # bumper-to-bumper distance
        distance_b2b_valid = max(0.001, distance_b2b)

        leading_v = leading_vehicle.get_state().v
        leading_v *= np.cos(leading_vehicle.get_state().theta - leading_transform.theta)
        current_v = self.get_state().v
        delta_v = current_v - leading_v
        
        s = current_v*(self.time_gap+delta_v/(2*np.sqrt(self.acceleration*self.comfortable_deceleration)))
        distance_desired = self.min_gap + max(0, s)

        v_rational = (current_v / self.desired_speed)**self.delta
        s_rational = (distance_desired / distance_b2b_valid) ** 2
        acceleration_target = self.acceleration * (1 - v_rational - s_rational)
        target_v = current_v + acceleration_target / self.decision_frequency


        # a = (distance_b2b, (self.min_gap + current_v*self.time_gap) / (1-(current_v/self.desired_speed)**self.delta)*0.5)
        # print(a, (target_v, current_v), (acceleration_target, 1-v_rational, s_rational) )
        # print()

        return target_v



def get_leading_vehicle(vehicle: Vehicle, vehicles: List[Vehicle], remaining_transforms: List[Transform], max_distance, scale_x=1.0, scale_y=1.0):
    """
        Get leading vehicle wrt remaining_transforms or global_path.
        !warning: distances between remaining_transforms cannot exceed any vehicle length.
    
    Args:
        remaining_transforms: list of Transform
    
    Returns:
        
    """
    
    current_transform = vehicle.get_transform()
    vehicle_vi = vehicle.vi
    vehicle_bbx = vehicle.bounding_box
    vehicle_half_length, vehicle_half_width = vehicle.bbx_x, vehicle.bbx_y
    func = lambda t: t.distance(current_transform)
    obstacles = [(func(o.get_transform()), o) for o in vehicles if o.vi != vehicle_vi and func(o.get_transform()) <= 1.001*max_distance]
    sorted_obstacles = sorted(obstacles, key=lambda x:x[0])

    leading_vehicle, leading_transform, leading_distance = None, None, 0.0
    for i, transform in enumerate(remaining_transforms):
        if i > 0:
            leading_distance += transform.distance(remaining_transforms[i-1])
        if leading_distance > 1.001*max_distance:
            break
        for _, obstacle in sorted_obstacles:
            obstacle_transform = obstacle.get_transform()
            obstacle_bbx = obstacle.bounding_box
            future_bbx = ActorBoundingBox(transform, vehicle_half_length *scale_x, vehicle_half_width *scale_y)
            # if obstacle_bbx.intersects(future_bbx) and not obstacle_bbx.intersects(vehicle_bbx):
            _longitudinal_e, _, _ = error_transform(current_transform, obstacle_transform)
            if obstacle_bbx.intersects(future_bbx) and _longitudinal_e < 0:  ### ! warning
                leading_vehicle = obstacle
                leading_transform = transform
                longitudinal_e, _, _ = error_transform(obstacle_transform, transform)
                leading_distance += longitudinal_e
                break
        if leading_vehicle != None:
            break
    return leading_vehicle, leading_transform, leading_distance




def side_transform(transform: Transform, half_width):
    center = np.array([transform.x, transform.y])
    theta = transform.theta
    direction = np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)])
    left, right = center + half_width * direction, center - half_width * direction
    return Transform(left[0], left[1], theta), Transform(right[0], right[1], theta)

