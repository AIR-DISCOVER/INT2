# INT2: INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import math
import numpy as np
from math import sqrt
import matplotlib.path as mpath
from IPython import embed

delta_d = 2.5
delta_move_d = 5
time_len_d = 5
scenario_min_len = 91
current_time_len = 11
min_displacement = 0.1
min_velocity = 0.1

# vehicle = 1, pedestrian = 2, cyclist = 3
OBJECT_TYPE_DICT = {
    "v2v": [2, 2], 
    "v2p": [2, 1], 
    "v2c": [2, 0]
}

TYPE_DICT = {
    "v": 2, 
    "p": 1, 
    "c": 0
}

def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y

class Normalizer:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.origin = rotate(0.0 - x, 0.0 - y, yaw)

    def __call__(self, points, reverse=False):
        points = np.array(points)
        if points.shape == (2,):
            points.shape = (1, 2)
        assert len(points.shape) <= 3
        if len(points.shape) == 3:
            for each in points:
                each[:] = self.__call__(each, reverse)
        else:
            assert len(points.shape) == 2
            for point in points:
                if reverse:
                    point[0], point[1] = rotate(point[0] - self.origin[0],
                                                point[1] - self.origin[1], -self.yaw)
                else:
                    point[0], point[1] = rotate(point[0] - self.x,
                                                point[1] - self.y, self.yaw)

        return points

def _get_normalized(polygons, x, y, angle):
    cos_ = math.cos(angle)
    sin_ = math.sin(angle)
    n = polygons.shape[1]

    new_polygons = np.zeros((polygons.shape[0], n, 2), dtype=np.float32)
    for polygon_idx in range(polygons.shape[0]):
        for i in range(n):
            polygons[polygon_idx, i, 0] -= x
            polygons[polygon_idx, i, 1] -= y
            new_polygons[polygon_idx, i, 0] = polygons[polygon_idx, i, 0] * cos_ - polygons[polygon_idx, i, 1] * sin_
            new_polygons[polygon_idx, i, 1] = polygons[polygon_idx, i, 0] * sin_ + polygons[polygon_idx, i, 1] * cos_

    return new_polygons

def get_normalized(trajectorys, normalizer, reverse=False):
    if isinstance(trajectorys, np.float32):
        trajectorys = trajectorys.astype(np.float32)

    if reverse:
        return _get_normalized(trajectorys, normalizer.origin[0], normalizer.origin[1], -normalizer.yaw)
    
    return _get_normalized(trajectorys, normalizer.x, normalizer.y, normalizer.yaw)



def confirm_interactive(inf_track_type, rea_track_type, direction, i2r_rel_position, intersect):

    if intersect == 'CROSS':
        return True

    co_direction_invalid_type_str_list = [
        'LEFT, STRAIGHT, STRAIGHT',
        'LEFT, STRAIGHT, RIGHT-TURN',
        'LEFT, STRAIGHT, STRAIGHT-RIGHT',
        'LEFT, LEFT-TURN, STRAIGHT',
        'LEFT, LEFT-TURN, RIGHT-TURN',
        'LEFT, STRAIGHT-LEFT, STRAIGHT',
        'LEFT, STRAIGHT-LEFT, STRAIGHT-RIGHT',
        
        'RIGHT, STRAIGHT, LEFT-TURN',
        'RIGHT, STRAIGHT, STRAIGHT',
        'RIGHT, STRAIGHT, STRAIGHT-LEFT',
        'RIGHT, RIGHT-TURN, STRAIGHT',
        'RIGHT, RIGHT-TURN, LEFT-TURN',
        'RIGHT, STRAIGHT-RIGHT, STRAIGHT'
        'RIGHT, STRAIGHT-RIGHT, STRAIGHT-LEFT',
    ]

    opp_direction_invalid_type_str_list = [
        'LEFT, STRAIGHT, STRAIGHT',
        'RIGHT, STRAIGHT, STRAIGHT',
    ]

    temp_str = ', '.join([i2r_rel_position, inf_track_type, rea_track_type])

    if direction == 'CO-DIRECTION':
        if temp_str in co_direction_invalid_type_str_list:
            return False
        else:
            return True
    
    if direction == 'OPP-DIRECTION':
        if temp_str in opp_direction_invalid_type_str_list:
            return False
        else:
            return True

    return True

def get_dis_point(a, b):
    return sqrt(a * a + b * b)

def classify_track(trajectory, n=None):
    end_state = -1
    history_frame_num = 11
    kMaxSpeedForStationary = 2.0
    kMaxDisplacementForStationary = 5.0
    kMaxLateralDisplacementForStraight = 5  # 5
    kMinLongitudinalDisplacementForUTurn = -5  # -5
    kMaxAbsHeadingDiffForStraight = math.pi / 6.0

    # start_state = history_frame_num - 1
    start_state = 0

    ## The difference in the horizontal coordinates between the starting
    ## point and the ending point of a vehicle's trajectory.
    x_delta = trajectory[end_state, 0] - trajectory[start_state, 0]
    y_delta = trajectory[end_state, 1] - trajectory[start_state, 1]

    ## The distance between the horizontal coordinates of the starting 
    ## point and the ending point of a vehicle's trajectory.
    final_displacement = get_dis_point(x_delta, y_delta)
    # 头部夹角的变化
    heading_diff = trajectory[end_state, 2] - trajectory[start_state, 2]
    cos_ = math.cos(-trajectory[start_state, 2])
    sin_ = math.sin(-trajectory[start_state, 2])
    x = x_delta
    y = y_delta
    dx, dy = x * cos_ - y * sin_, x * sin_ + y * cos_
    start_speed = get_dis_point(trajectory[start_state, 3], trajectory[start_state, 4])
    end_speed = get_dis_point(trajectory[end_state, 3], trajectory[end_state, 4])
    max_speed = max(start_speed, end_speed)

    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return 'STATIONARY'
    if abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if abs(dy) < kMaxLateralDisplacementForStraight:
            return 'STRAIGHT'
        return 'STRAIGHT-RIGHT' if dy < 0 else 'STRAIGHT-LEFT'

    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return 'RIGHT-TURN'
        # return 'RIGHT_U_TURN' if dx < kMinLongitudinalDisplacementForUTurn else 'RIGHT_TURN'
    if dx < kMinLongitudinalDisplacementForUTurn:
        return 'LEFT-U-TURN'

    return 'LEFT-TURN'


def i2r_relative_position(trajectorys):
    kMaxDistanceForPosition = 0
    kMaxDistanceForSameLine = 1
    RateDirectionLeftRight = 0.5
    RatePositionLeftRight = 0.6
    # kMaxDistanceForVertical = 3
    kMaxSubDeltaXYFordirection = 5
    # rotate
    normalizer = Normalizer(trajectorys[0, 0, 0], trajectorys[0, 0, 1], -trajectorys[0, 0, 2] + math.radians(90))
    position_xy = get_normalized(trajectorys, normalizer)
    inf_x = position_xy[0, :, 0]
    inf_y = position_xy[0, :, 1]
    rea_x = position_xy[1, :, 0]
    rea_y = position_xy[1, :, 1]

    sub_x = inf_x - rea_x
    r_sub_y = rea_y[1:] - rea_y[:-1]

    reactor_direction = None
    position  = None
    intersect = None
    
    rea_delta_x = abs(rea_x[1:10] - rea_x[0:9])
    rea_delta_y = abs(rea_y[1:10] - rea_y[0:9])

    if sum(rea_delta_x - rea_delta_y) > kMaxSubDeltaXYFordirection:
        reactor_direction = 'VERTICAL'
    else:
        if np.sum(r_sub_y > 0) >= len(r_sub_y) * RateDirectionLeftRight:
            reactor_direction = 'CO-DIRECTION'
        else:
            reactor_direction = 'OPP-DIRECTION'

    if abs(np.median(sub_x)) < kMaxDistanceForSameLine:
        position = 'SAME-LINE'
    elif np.sum(sub_x > 0) > len(sub_x) * RatePositionLeftRight:
        position = 'RIGHT'
    elif np.sum(sub_x < 0) > len(sub_x) * RatePositionLeftRight:
        position = 'LEFT'
    else:
        if np.sum(sub_x > 0) >= np.sum(sub_x < 0):
            position = 'RIGHT'
        else:
            position = 'LEFT'
    
    intersect = is_intersect(position_xy)
    
    return reactor_direction, position, intersect

def is_intersect(trajectorys):
    i_point_x = trajectorys[0, :, 0]
    i_point_y = trajectorys[0, :, 1]
    r_point_x = trajectorys[1, :, 0]
    r_point_y = trajectorys[1, :, 1]
    path_i = mpath.Path(list(zip(i_point_x, i_point_y)))
    path_r = mpath.Path(list(zip(r_point_x, r_point_y)))
    if path_i.intersects_path(path_r):
        return 'CROSS'
    return 'UNCROSSED'

def is_interaction_valid(i, j, agent_i_info, agent_j_info):
    agent_i_x = agent_i_info[0]
    agent_i_y = agent_i_info[1]
    agent_i_w = agent_i_info[2]
    agent_i_l = agent_i_info[3]
    agent_i_t = agent_i_info[4]
    agent_i_vx = agent_i_info[5]
    agent_i_vy = agent_i_info[6]

    agent_j_x = agent_j_info[0]
    agent_j_y = agent_j_info[1]
    agent_j_w = agent_j_info[2]
    agent_j_l = agent_j_info[3]
    agent_j_t = agent_j_info[4]
    agent_j_vx = agent_j_info[5]
    agent_j_vy = agent_j_info[6]

    # Remove static agents
    agent_i_distance = np.sum(np.sqrt((agent_i_x[-1] - agent_i_x[0]) ** 2 + (agent_i_y[-1] - agent_i_y[0]) ** 2))
    if agent_i_distance < delta_move_d:
        return False, None, None
    agent_j_distance = np.sum(np.sqrt((agent_j_x[-1] - agent_j_x[0]) ** 2 + (agent_j_y[-1] - agent_j_y[0]) ** 2))
    if agent_j_distance < delta_move_d:
        return False, None, None

    # Filter out non-interacting pairs based on spatiotemporal distance threshold.
    distance = np.sqrt((agent_i_x.reshape(-1, 1) - agent_j_x) ** 2 + (agent_i_y.reshape(-1, 1) - agent_j_y) ** 2)
    d_min = distance.min()

    agent_i_min_distance_idx = (distance == d_min).nonzero()[0][0]
    agent_j_min_distance_idx = (distance == d_min).nonzero()[1][0]

    distance_true = np.diagonal(distance)
    time_step_distance_argmin = np.argmin(distance_true)

    ## If the distance to the nearest frame is greater than 11, discard it directly.
    if time_step_distance_argmin < current_time_len:
        return False, None, None
    
    threshold_value = (sqrt(agent_i_w[time_step_distance_argmin]**2 + agent_i_l[time_step_distance_argmin]**2) \
        + sqrt(agent_j_w[time_step_distance_argmin]**2 + agent_j_l[time_step_distance_argmin]**2)) / 2 + delta_d

    ## Obtain pairs that may have interactions based on a distance threshold.
    if distance_true[time_step_distance_argmin] >= threshold_value:
        return False, None, None

    threshold_value_all = (np.sqrt(agent_i_w**2 + agent_i_l**2) + np.sqrt(agent_j_w**2 + agent_j_l**2)) / 2 + delta_d
    interaction_time_valid = distance_true < threshold_value_all
    
    # filter pairs when thedistance is so closest and moving snow even stati
    agent_i_displacement = np.sqrt((agent_i_x[1:] - agent_i_x[:-1]) ** 2 + (agent_i_y[1:] - agent_i_y[:-1]) ** 2)
    agent_j_displacement = np.sqrt((agent_j_x[1:] - agent_j_x[:-1]) ** 2 + (agent_j_y[1:] - agent_j_y[:-1]) ** 2)
    agent_i_velocity = np.sqrt(agent_i_vx ** 2 + agent_i_vy ** 2)
    agent_j_velocity = np.sqrt(agent_j_vx ** 2 + agent_j_vy ** 2)

    agent_i_displacement_valid = list(agent_i_displacement > min_displacement)
    agent_j_displacement_valid = list(agent_j_displacement > min_displacement)
    agent_i_displacement_valid.insert(agent_i_displacement_valid[0], 0)
    agent_j_displacement_valid.insert(agent_j_displacement_valid[0], 0)
    agent_i_displacement_valid = np.array(agent_i_displacement_valid)
    agent_j_displacement_valid = np.array(agent_j_displacement_valid)
    agent_i_velocity_valid = agent_i_velocity > min_velocity
    agent_j_velocity_valid = agent_j_velocity > min_velocity

    agent_i_j_valid_time = np.logical_and(np.logical_and(agent_i_displacement_valid, agent_i_velocity_valid), 
                                          np.logical_and(agent_j_displacement_valid, agent_j_velocity_valid))
    
    interaction_time_valid = np.logical_and(interaction_time_valid, agent_i_j_valid_time)

    if interaction_time_valid.sum() < 5:  # Avoid outliers
        return False, None, None

    relation_type = 0
    if agent_i_min_distance_idx < agent_j_min_distance_idx:
        relation_type = 0
    elif agent_i_min_distance_idx > agent_j_min_distance_idx:
        relation_type = 1
    else:
        
        ### agent_i_min_distance_idx == agent_j_min_distance_idx
        ### Calculate the distance from the previous frame that is closest in distance to the current frame. 
        ### The closest frame is considered the leading vehicle.
        pre_dis_i = sqrt((agent_i_x[time_step_distance_argmin] - agent_i_x[time_step_distance_argmin - 1]) ** 2 \
                        + (agent_i_y[time_step_distance_argmin] - agent_i_y[time_step_distance_argmin - 1]) ** 2)
        
        pre_dis_j = sqrt((agent_j_x[time_step_distance_argmin] - agent_j_x[time_step_distance_argmin - 1]) ** 2 \
                        + (agent_j_y[time_step_distance_argmin] - agent_j_y[time_step_distance_argmin - 1]) ** 2)


        if pre_dis_i < pre_dis_j:
            relation_type = 0
        else:
            relation_type = 1

    i_trajectory = np.stack([agent_i_x, agent_i_y, agent_i_t, agent_i_vx, agent_i_vy], axis=1)
    j_trajectory = np.stack([agent_j_x, agent_j_y, agent_j_t, agent_j_vx, agent_j_vy], axis=1)

    i_track_type = classify_track(i_trajectory)
    j_track_type = classify_track(j_trajectory)

    influencer_track_type = None
    reactor_track_type = None
    if relation_type == 0:
        influencer_track_type = i_track_type
        reactor_track_type = j_track_type
        direction, i2r_rel_position, intersect = i2r_relative_position(np.stack([i_trajectory, j_trajectory], axis=0))
    else:
        influencer_track_type = j_track_type
        reactor_track_type = i_track_type
        direction, i2r_rel_position, intersect = i2r_relative_position(np.stack([j_trajectory, i_trajectory], axis=0))

    if confirm_interactive(influencer_track_type, reactor_track_type, direction, i2r_rel_position, intersect):
        return True, relation_type, interaction_time_valid
    
    return False, None, None

