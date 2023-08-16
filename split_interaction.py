# INT2: INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import os
import numpy as np
from math import sqrt
import pickle
from utils.interaction_utils import *
import warnings
import argparse
warnings.filterwarnings("ignore")
from p_tqdm import p_map

def parse_config():
    parser = argparse.ArgumentParser(description='INT2 Dataset Interaction Filter Visualization.')
    parser.add_argument('--interaction_scenario_path', type=str, default='int2_dataset_example/interaction_scenario/complete_scenario/8/010213355106-010213364106.pickle',
                         help='The scenario path to be visualized')
    parser.add_argument('--output_dir', type=str, default='int2_dataset_example/interaction_scenario/split_scenario', help='')
    args = parser.parse_args()

    return args

def split_interaction(interaction_scenario_path, out_dir):
    hdmap_id = interaction_scenario_path.split('/')[-2]
    output_dir = os.path.join(out_dir, hdmap_id)
    os.makedirs(output_dir, exist_ok=True)

    with open(interaction_scenario_path, 'rb+') as f:
        interaciton_scenario_info = pickle.load(f)

    state = interaciton_scenario_info['AGENT_INFO']['state']
    position_x = state['position_x']
    position_y = state['position_y']
    velocity_x = state['velocity_x']
    velocity_y = state['velocity_y']
    INTERACTION_INFO = interaciton_scenario_info['INTERACTION_INFO']
    
    interaction_pair_info = INTERACTION_INFO['interaction_pair_info']
    interaction_info_new = {}
    n = 0

    inter_pairs_num = len(interaction_pair_info.keys())
    interaction_complete_list = []

    for i in range(0, inter_pairs_num - 1, 1):

        relation_set = set()
        relation_set.add(interaction_pair_info[i]['influencer_id'])
        relation_set.add(interaction_pair_info[i]['reactor_id'])
        
        coexistence_time = [int(x) for x in interaction_pair_info[i]['coexistence_time']]
        interaction_time = [int(x) for x in interaction_pair_info[i]['interaction_time']]
        for j in range(i + 1, inter_pairs_num, 1):

            now_interaction_info = interaction_pair_info[j]

            now_i_id = now_interaction_info['influencer_id']
            now_r_id = now_interaction_info['reactor_id']
            now_co_time = [int(x) for x in now_interaction_info['coexistence_time']]
            now_inter_time = [int(x) for x in now_interaction_info['interaction_time']]
            if (now_i_id in relation_set) and (now_r_id in relation_set):
                continue

            if (now_i_id in relation_set) or (now_r_id in relation_set):
                coexistence_time_new = [x for x in now_co_time if x in coexistence_time]
                interaction_time_new = [x for x in now_inter_time if x in interaction_time]

                if len(coexistence_time_new) > scenario_min_len and len(interaction_time_new) > 0:
                    relation_set.add(now_i_id)
                    relation_set.add(now_r_id)
                    coexistence_time = coexistence_time_new
                    interaction_time = interaction_time_new

        relation_list = list(sorted(relation_set))

        if relation_list in interaction_complete_list:
            continue

        interaction_complete_list.append(relation_list)
        interaction_info_new[n] = {
            'relation_list' : relation_list,
            'coexistence_time': coexistence_time,
            'interaction_time': interaction_time
        }
        n += 1
    
    need2delete_indices = []
    for i in range(0, len(interaction_complete_list) - 1, 1):
        for j in range(i + 1, len(interaction_complete_list), 1):
            inner_list = [x for x in interaction_complete_list[i] if x in interaction_complete_list[j]]
            if len(inner_list) == len(interaction_complete_list[i]):
                need2delete_indices.append(i)
                break
            elif len(inner_list) == len(interaction_complete_list[j]):
                need2delete_indices.append(j)
                break
            else:
                pass
    need2delete_indices = list(set(need2delete_indices))
    for key in need2delete_indices:
        interaction_info_new.pop(key)

        
    result = {}
    tmp = 0
    for key, value in interaction_info_new.items():
        coexistence_start = value['coexistence_time'][0]
        coexistence_end = value['coexistence_time'][-1]

        interaction_time = np.array(value['interaction_time'])
        interaction_start = interaction_time[0]
        interaction_end = interaction_time[-1]

        interaction_len = interaction_end - interaction_start + 1

        if interaction_len > 91:
            max_coverage = 0
            best_start = 0
            best_end = 0

            for start_frame in interaction_time:
                end_frame = start_frame + 90
                coverage_valid = np.logical_and(interaction_time >= start_frame, interaction_time <= end_frame)
                coverage_num = coverage_valid.sum()
                if coverage_num > max_coverage:
                    max_coverage = coverage_num
                    best_start = start_frame
                    best_end = interaction_time[coverage_valid][-1]

            interaction_start = best_start
            interaction_end = best_end

        start_r = max(interaction_start - 20, coexistence_start)
        end_r = start_r + 90
        if end_r < interaction_end:
            diff = interaction_end - end_r
            end_r += diff
            start_r += diff

        if end_r > coexistence_end:
            diff = end_r - coexistence_end
            end_r -= diff
            start_r -= diff
        
        result[tmp] = {
            'interested_agents': value['relation_list'],
            'split_time_91': [start_r, end_r],
        }
        tmp += 1

    output_path = os.path.join(output_dir, interaction_scenario_path.split('/')[-1])
    with open(output_path, 'wb+') as f:
        f.write(pickle.dumps(result))

    # print(output_path)

def main():
    args = parse_config()
    assert args.interaction_scenario_path != None
    assert args.output_dir != None
    split_interaction(args.interaction_scenario_path, args.output_dir)

def multi_process():
    data_dir = 'int2_dataset/interaction_scenario/complete_scenario'
    output_dir = 'int2_dataset/interaction_scenario/split_scenario'
    dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    for d_dir in dirs:
        print(f'process in {d_dir}')
        file_list = [os.path.join(d_dir, f) for f in os.listdir(os.path.join(d_dir))]
        p_map(split_interaction, file_list, [output_dir] * len(file_list), num_cpus=0.2)

if __name__ == "__main__":
    main()
    # multi_process()