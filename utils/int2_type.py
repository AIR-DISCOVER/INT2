# INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

tf_state_map = {
    0: 'RED',
    1: 'GREEN',
    2: 'YELLOW'
}

agent_type_map = {
    0: 'BICYCLE',
    1: 'PEDESTRIAN',
    2: 'VEHICLE'
}

agent_sub_type_map = {
    0: 'CYCLIST',
    1: 'MOTORCYCLIST',
    2: 'TRICYCLIST',
    3: 'PEDESTRIAN',
    4: 'CAR',
    5: 'VAN',
    6: 'BUS',
    7: 'TRUCK'
}

lane_type_map = {
    0: 'BIKING',
    1: 'CITY_DRIVING',
    2: 'EMERGENCY_LANE',
    3: 'LEFT_TURN_WAITING_ZONE',
    4: 'ROUNDABOUT'
}