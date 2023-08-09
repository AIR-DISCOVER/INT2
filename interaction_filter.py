# INT2: INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import pandas as pd
import numpy as np
from math import sqrt
import os
import json
import sys
import copy
import time
import copy
import math
import re
import tqdm
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from p_tqdm import p_map
from IPython import embed
import warnings
import argparse
warnings.filterwarnings("ignore")
from utils.interaction_utils import *

def parse_config():
    parser = argparse.ArgumentParser(description='INT2 Dataset Interaction Filter Visualization.')
    parser.add_argument('--scenario_path', type=str, default='int2_dataset_example/scenario/0/010213250706-010213264206.pickle',
                         help='The scenario path to be visualized')
    parser.add_argument('--output_dir', type=str, default='int2_dataset_example/interaction_scenario/complete_scenario', help='')
    args = parser.parse_args()

    return args

def interaction_define(scenario_path, output_dir):
    with open(scenario_path, 'rb+') as f:
        scenario_info = pickle.load(f)
    AGENT_INFO = scenario_info['AGENT_INFO']

    object_id = AGENT_INFO['object_id']
    object_type = AGENT_INFO['object_type']
    object_sub_type = AGENT_INFO['object_sub_type']
    state = AGENT_INFO['state']
    interaction_info = {}
    
    position_x = state['position_x']
    position_y = state['position_y']
    position_z = state['position_z']
    theta      = state['theta']
    velocity_x = state['velocity_x']
    velocity_y = state['velocity_y']
    length     = state['length']
    width      = state['width']
    height     = state['height']
    valid      = state['valid']
    
    inter_info_dict = {}
    inter_pair_info_dict = {}
    inter_pair_index = 0
    agent_num = valid.shape[0]
    ir_indices_list = []

    for i in range(0, agent_num - 1, 1):
        if object_type[i] != 2:
            continue
        valid_i = valid[i].nonzero()[0]

        if len(valid_i) < scenario_min_len:
            continue
        for j in range(i + 1, agent_num, 1):
            valid_j = valid[j].nonzero()[0]
            if len(valid_j) < scenario_min_len:
                continue
                
            coexistence_time = np.array([x for x in valid_i if x in valid_j])
             
            if len(coexistence_time) < scenario_min_len:
                continue
            
            agent_i_x = position_x[i][coexistence_time]
            agent_i_y = position_y[i][coexistence_time]
            agent_i_w = width[i][coexistence_time]
            agent_i_l = length[i][coexistence_time]
            agent_i_t = theta[i][coexistence_time]
            agent_i_vx = velocity_x[i][coexistence_time]
            agent_i_vy = velocity_y[i][coexistence_time]
            agent_i_info = np.stack([agent_i_x, agent_i_y, agent_i_w, agent_i_l, agent_i_t, agent_i_vx, agent_i_vy], axis=0)

            agent_j_x = position_x[j][coexistence_time]
            agent_j_y = position_y[j][coexistence_time]
            agent_j_w = width[j][coexistence_time]
            agent_j_l = length[j][coexistence_time]
            agent_j_t = theta[j][coexistence_time]
            agent_j_vx = velocity_x[j][coexistence_time]
            agent_j_vy = velocity_y[j][coexistence_time]
            agent_j_info = np.stack([agent_j_x, agent_j_y, agent_j_w, agent_j_l, agent_j_t, agent_j_vx, agent_j_vy], axis=0)

            inter_is_ok, relation_type, interaction_time_valid = is_interaction_valid(i, j, agent_i_info, agent_j_info)
            if inter_is_ok:
                interaction_time_truth = np.array(coexistence_time)[interaction_time_valid]
                if relation_type == 0:
                    influencer_id = i
                    reactor_id = j
                else:
                    influencer_id = j
                    reactor_id = i
                ir_indices_list.append([influencer_id, reactor_id])
                inter_pair_info_dict[inter_pair_index] = {
                    'influencer_id': influencer_id,
                    'reactor_id': reactor_id,
                    'influencer_type': object_type[influencer_id],
                    'reactor_type': object_type[reactor_id],
                    'coexistence_time': coexistence_time,
                    'interaction_time': interaction_time_truth
                }

                inter_pair_index += 1

    interested_agents = set()
    for i in range(len(ir_indices_list)):
        interested_agents.add(ir_indices_list[i][0])
        interested_agents.add(ir_indices_list[i][1])
    inter_info_dict['interaction_pair_info'] = inter_pair_info_dict
    inter_info_dict['interested_agents'] = list(interested_agents)       
    output_path = os.path.join(output_dir, scenario_path.split('/')[-1])

    scenario_info['INTERACTION_INFO'] = inter_info_dict
    
    with open(output_path, 'wb+') as f:
        f.write(pickle.dumps(scenario_info))
    
    # print(output_path)


def single_process(scenario_path, output_floder):
    error_scenario_path = 'error_scenario.txt'
    try:
        hdmap_id = scenario_path.split('/')[-2]
        output_dir = os.path.join(output_floder, hdmap_id)
        os.makedirs(output_dir, exist_ok=True)
        interaction_define(scenario_path, output_dir)
    except:
        with open(error_scenario_path, 'a') as f:
            f.write(scenario_path + '\n')

def multi_thread_process():
    scenario_floder = 'int2_dataset/scenario'
    output_floder = 'int2_dataset/interaction_scenario/complete_scenario'
    scenario_dir_names = sorted(os.listdir(scenario_floder), key=lambda x: int(x))
    
    for idx, scenario_id in enumerate(scenario_dir_names):
        if idx < 13:
            continue
        print(f'now are processed in {scenario_id}th')
        scenario_files = [os.path.join(scenario_floder, scenario_id, f) for f 
                          in os.listdir(os.path.join(scenario_floder, scenario_id))]
        
        p_map(single_process, scenario_files, [output_floder] * len(scenario_files), num_cpus=0.2)


def main():
    # multi_thread_process()
    args = parse_config()
    assert args.scenario_path != None
    assert args.output_dir != None
    single_process(args.scenario_path, args.output_dir)

if __name__ == "__main__":
    main()