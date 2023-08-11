# INT2: INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import sys, os
import numpy as np
import pickle
import tensorflow as tf
import multiprocessing
import glob
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from IPython import embed
from p_tqdm import p_map
import pickle

def format_preprocess(complete_scenario_path):   
    split_scenario_path = complete_scenario_path.replace('complete_scenario', 'split_scenario')
    assert os.path.exists(split_scenario_path) and os.path.exists(complete_scenario_path)
    complete_scenario_data = None
    split_scenario_data = None
    with open(complete_scenario_path, 'rb+') as f:
        complete_scenario_data = pickle.load(f)
    with open(split_scenario_path, 'rb+') as f:
        split_scenario_data = pickle.load(f)
    
    split_path = complete_scenario_path.split('/')
    data_dir = '/'.join(split_path[:3])
    hdmap_id = split_path[-2]
    hdmap_path = os.path.join(data_dir, 'hdmap', hdmap_id + '.json')
    embed()

    agent_info = complete_scenario_data['AGENT_INFO']
    traffic_lights_info = complete_scenario_data['TRAFFIC_LIGHTS_INFO']
    timestame_scenario = complete_scenario_data['TIMESTAMP_SCENARIO'].astype(np.float64)
    object_id = agent_info['object_id']
    object_type = agent_info['object_type']
    object_sub_type = agent_info['object_sub_type']
    state = agent_info['state']

    tf_state = traffic_lights_info['tf_state']
    tf_state_valid = traffic_lights_info['tf_state_valid']
    tf_mapping_lane_id = traffic_lights_info['tf_mapping_lane_id']
    
    '''
    position_x, position_y, position_z, theta, velocity_x, velocity_y, length, width, height, valid
    ''' 
    final_data_sample_list = {}
    split_scenario_num = len(split_scenario_data.keys())
    for index in range(split_scenario_num):
        interested_agents = split_scenario_data[index]['interested_agents']
        assert len(interested_agents) >= 2
        split_time_91 = split_scenario_data[index]['split_time_91']
        assert split_time_91[1] - split_time_91[0] + 1 == 91
        for i in range(0, len(interested_agents) - 1):
            int_0_idx = interested_agents[i]
            agent_0_x = state['position_x'][int_0_idx, split_time_91[0]:split_time_91[1] + 1]
            agent_0_y = state['position_y'][int_0_idx, split_time_91[0]:split_time_91[1] + 1]
            inf_idx = 0
            rea_idx = 0

            valid     = state['valid'][:, split_time_91[0]:split_time_91[1] + 1]
            agent_valid_idx = np.where(valid.sum(axis=1) != 0)[0]
            agent_num = len(agent_valid_idx)

            bbox_yaw        = state['theta'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            x               = state['position_x'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            y               = state['position_y'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            z               = state['position_z'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            h               = state['height'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            w               = state['width'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            l               = state['height'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            v_x             = state['velocity_x'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            v_y             = state['velocity_y'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx]
            vel_way         = np.arctan(v_y / (v_x + 1e-9))
            timestame       = timestame_scenario[split_time_91[0]:split_time_91[1] + 1]
            object_id       = object_id[agent_valid_idx]
            object_type     = object_type[agent_valid_idx]
            object_sub_type = object_sub_type[agent_valid_idx]

            for j in range(i + 1, len(interested_agents)):
                int_1_idx = interested_agents[j]
                assert int_0_idx in agent_valid_idx and int_1_idx in agent_valid_idx
                agent_1_x = state['position_x'][int_1_idx, split_time_91[0]:split_time_91[1] + 1]
                agent_1_y = state['position_y'][int_1_idx, split_time_91[0]:split_time_91[1] + 1]
                distance = np.sqrt((agent_0_x.reshape(-1, 1) - agent_1_x) ** 2 + (agent_0_y.reshape(-1, 1) - agent_1_y) ** 2)
                d_min = distance.min()
                agent_0_min_dis_idx = (distance == d_min).nonzero()[0][0]
                agent_1_min_dis_idx = (distance == d_min).nonzero()[1][0]

                distance_true = np.diagonal(distance)
                time_step_distance_argmin = np.argmin(distance_true)
                
                if agent_0_min_dis_idx < agent_1_min_dis_idx:
                    inf_idx = np.where(agent_valid_idx == int_0_idx)[0].item()
                    rea_idx = np.where(agent_valid_idx == int_1_idx)[0].item()
                elif agent_0_min_dis_idx > agent_1_min_dis_idx:
                    inf_idx = np.where(agent_valid_idx == int_1_idx)[0].item()
                    rea_idx = np.where(agent_valid_idx == int_0_idx)[0].item()
                else:
                    pre_dis_i = np.sqrt((agent_0_x[time_step_distance_argmin] - agent_0_x[time_step_distance_argmin - 1]) ** 2 \
                                    + (agent_0_y[time_step_distance_argmin] - agent_0_y[time_step_distance_argmin - 1]) ** 2)
        
                    pre_dis_j = np.sqrt((agent_1_x[time_step_distance_argmin] - agent_1_x[time_step_distance_argmin - 1]) ** 2 \
                                    + (agent_1_y[time_step_distance_argmin] - agent_1_y[time_step_distance_argmin - 1]) ** 2)
                    
                    if pre_dis_i < pre_dis_j:
                        inf_idx = np.where(agent_valid_idx == int_0_idx)[0].item()
                        rea_idx = np.where(agent_valid_idx == int_1_idx)[0].item()
                    else:
                        inf_idx = np.where(agent_valid_idx == int_1_idx)[0].item()
                        rea_idx = np.where(agent_valid_idx == int_0_idx)[0].item()

                objects_of_interest = 0 * np.ones((agent_num), dtype=np.int32)
                tracks_to_predict = 0 * np.ones((agent_num), dtype=np.int32)
                is_sdc = -1 * np.ones((agent_num), dtype=np.int32)
                
                objects_of_interest[inf_idx] = 1
                objects_of_interest[rea_idx] = 1
                tracks_to_predict[inf_idx] = 1
                tracks_to_predict[rea_idx] = 1
                is_sdc[inf_idx] = 1

                embed()

                decoded_example = {}


                


def main():
    data_dir = '../int2_dataset/interaction_scenario'
    complete_scenario_path = os.path.join(data_dir, 'complete_scenario')
    dirs = [os.path.join(complete_scenario_path, f) for f in os.listdir(complete_scenario_path)]
    for data_dir in dirs:
        complete_scenario_path_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        format_preprocess(complete_scenario_path_list[0])



if __name__ == "__main__":
    main()