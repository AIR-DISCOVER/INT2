# INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import os
import numpy as np
import pickle
from p_tqdm import p_map
import pickle
import json
import copy
from IPython import embed

tf_state_match = {
    0: 4,
    1: 6,
    2: 5,
}

agent_type_match = {
    0: 3,
    1: 2,
    2: 1,
}

agent_sub_type_match = {
    0: 5,
    1: 6,
    2: 7,
    3: 8,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
}

lane_type_match = {
    0: 3,
    1: 2,
    2: 2,
    3: 17,
    4: 4
}

def agent_pair_lable_map(type1, type2):
    if type1 == 1 and type2 == 1:
        return 1
    elif (type1 == 1 and type2 == 2) or (type1 == 2 and type2 == 1):
        return 2
    elif (type1 == 1 and type2 == 3) or (type1 == 3 and type2 == 1):
        return 3
    else:
        assert 1 == 0

def scenario_format_preprocess(complete_scenario_path, output_dir):   
    split_scenario_path = complete_scenario_path.replace('complete_scenario', 'split_scenario')
    assert os.path.exists(split_scenario_path) and os.path.exists(complete_scenario_path)
    complete_scenario_data = None
    split_scenario_data = None
    with open(complete_scenario_path, 'rb+') as f:
        complete_scenario_data = pickle.load(f)

    with open(split_scenario_path, 'rb+') as f:
        try:
            split_scenario_data = pickle.load(f)
        except:
            # with open(os.path.join(output_dir, 'error.txt'), 'a') as e_f:
            #     e_f.write(f'{complete_scenario_path}\n')
            return

    split_path = complete_scenario_path.split('/')
    data_dir = '/'.join(split_path[:2])
    hdmap_id = split_path[-2]
    hdmap_path = os.path.join(data_dir, 'hdmap', hdmap_id + '.json')

    with open(hdmap_path, 'r') as f:
        hdmap_data = json.load(f)
    lane_info = hdmap_data['LANE']
    
    agent_info = complete_scenario_data['AGENT_INFO']
    traffic_lights_info = complete_scenario_data['TRAFFIC_LIGHTS_INFO']
    timestame_scenario = complete_scenario_data['TIMESTAMP_SCENARIO'].astype(np.float32)
    scenario_id = complete_scenario_data['SCENARIO_ID']
    object_id = agent_info['object_id']
    object_type = agent_info['object_type']
    object_sub_type = agent_info['object_sub_type']
    state = agent_info['state']

    tf_mapping_lane_id = traffic_lights_info['tf_mapping_lane_id']

    tf_position = []
    for tf_id in tf_mapping_lane_id:
        tf_mapping_lane_info = lane_info[f'{tf_id}']
        centerline = np.array(tf_mapping_lane_info['centerline'], dtype=np.float32)
        tf_position.append([centerline[0, 0], centerline[0, 1], 0])

    tf_position = np.array(tf_position)

    data_sample_list = []
    split_scenario_num = len(split_scenario_data.keys())
    for index in range(split_scenario_num):
        interested_agents = split_scenario_data[index]['interested_agents']
        assert len(interested_agents) >= 2
        split_time_91 = split_scenario_data[index]['split_time_91']
        assert split_time_91[1] - split_time_91[0] + 1 == 91

        valid = state['valid'][:, split_time_91[0]:split_time_91[1] + 1]
        agent_valid_idx = np.where(valid.sum(axis=1) != 0)[0]
        valid = valid[agent_valid_idx]
        agent_num = len(agent_valid_idx)

        bbox_yaw        = state['theta'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        x               = state['position_x'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        y               = state['position_y'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        z               = state['position_z'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        h               = state['height'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        w               = state['width'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        l               = state['height'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        v_x             = state['velocity_x'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        v_y             = state['velocity_y'][:, split_time_91[0]:split_time_91[1] + 1][agent_valid_idx].astype(np.float32)
        vel_way         = np.arctan(v_y / (v_x + 1e-9)).astype(np.float32)
        timestame       = timestame_scenario[split_time_91[0]:split_time_91[1] + 1].astype(np.float32)
        object_id_new       = object_id[agent_valid_idx]
        object_type_new     = np.array([agent_type_match[a_type] for a_type in object_type[agent_valid_idx]])
        object_sub_type_new = np.array([agent_sub_type_match[a_sub_type] for a_sub_type in object_sub_type[agent_valid_idx]])
        
        tf_id = tf_mapping_lane_id
        tf_state = traffic_lights_info['tf_state'][:, split_time_91[0]:split_time_91[1] + 1]
        tf_valid = traffic_lights_info['tf_state_valid'][:, split_time_91[0]:split_time_91[1] + 1]
        
        tf_state_new = -1 * np.zeros_like(tf_state)
        for tf_idx0 in range(tf_valid.shape[0]):
            for tf_idx1 in range(tf_valid.shape[1]):
                tf_state_new[tf_idx0, tf_idx1] = tf_state_match[tf_state[tf_idx0, tf_idx1]] if tf_valid[tf_idx0, tf_idx1] else -1

        tf_m2i_id    = np.array(tf_id)[np.newaxis, :].repeat(91, axis=0).astype(np.int32)
        tf_m2i_state = np.transpose(tf_state_new, [1, 0]).astype(np.int32)
        tf_m2i_valid = np.transpose(tf_valid, [1, 0]).astype(np.int32)
        tf_m2i_x     = tf_position[:, 0][np.newaxis, :].repeat(91, axis=0).astype(np.float32)
        tf_m2i_y     = tf_position[:, 1][np.newaxis, :].repeat(91, axis=0).astype(np.float32)
        tf_m2i_z     = tf_position[:, 2][np.newaxis, :].repeat(91, axis=0).astype(np.float32)

        timestamp_micros = timestame[np.newaxis, :].repeat(agent_num, axis=0)

        decoded_example = {
            'state/bbox_yaw': bbox_yaw, 
            'state/x': x, 
            'state/y': y, 
            'state/z': z, 
            'state/height': h, 
            'state/width': w, 
            'state/length': l, 
            'state/vel_yaw': vel_way, 
            'state/valid':  valid, 
            'state/velocity_x': v_x, 
            'state/velocity_y': v_y, 
            'state/timestamp_micros': timestamp_micros, 
            'state/id': object_id_new, 
            'state/type': object_type_new,      
            'state/s_type': object_sub_type_new,

            'traffic_light_state/id': tf_m2i_id, 
            'traffic_light_state/state': tf_m2i_state, 
            'traffic_light_state/valid': tf_m2i_valid,
            'traffic_light_state/x': tf_m2i_x, 
            'traffic_light_state/y': tf_m2i_y, 
            'traffic_light_state/z': tf_m2i_z,
        }
        
        for i in range(0, len(interested_agents) - 1):
            int_i_idx = np.where(agent_valid_idx == interested_agents[i] )[0].item()
            for j in range(i + 1, len(interested_agents)):
                int_j_idx = np.where(agent_valid_idx == interested_agents[j])[0].item()
                
                agent_i_type = object_type_new[int_i_idx]
                agent_j_type = object_type_new[int_j_idx]

                if agent_i_type != 1 and agent_j_type != 1:
                    continue
                
                agent_pair_label = agent_pair_lable_map(agent_i_type, agent_j_type)

                inf_idx = 0
                rea_idx = 0
                agent_i_x = x[int_i_idx]
                agent_i_y = y[int_i_idx]
                agent_j_x = x[int_j_idx]
                agent_j_y = y[int_j_idx]

                distance = np.sqrt((agent_i_x.reshape(-1, 1) - agent_j_x) ** 2 + (agent_i_y.reshape(-1, 1) - agent_j_y) ** 2)
                d_min = distance.min()
                agent_i_min_dis_idx = (distance == d_min).nonzero()[0][0]
                agent_j_min_dis_idx = (distance == d_min).nonzero()[1][0]

                distance_true = np.diagonal(distance)
                time_step_distance_argmin = np.argmin(distance_true)
                
                relation_id = 0
                # get inf_idx and rea_idx
                if agent_i_min_dis_idx < agent_j_min_dis_idx:
                    inf_idx = int_i_idx
                    rea_idx = int_j_idx
                    relation_id = 0
                elif agent_i_min_dis_idx > agent_j_min_dis_idx:
                    inf_idx = int_j_idx
                    rea_idx = int_i_idx
                    relation_id = 1
                else:
                    pre_dis_i = np.sqrt((agent_i_x[time_step_distance_argmin] - agent_i_x[time_step_distance_argmin - 1]) ** 2 \
                                    + (agent_i_y[time_step_distance_argmin] - agent_i_y[time_step_distance_argmin - 1]) ** 2)
        
                    pre_dis_j = np.sqrt((agent_j_x[time_step_distance_argmin] - agent_j_x[time_step_distance_argmin - 1]) ** 2 \
                                    + (agent_j_y[time_step_distance_argmin] - agent_j_y[time_step_distance_argmin - 1]) ** 2)
                    
                    if pre_dis_i < pre_dis_j:
                        inf_idx = int_i_idx
                        rea_idx = int_j_idx
                        relation_id = 0
                    else:

                        inf_idx = int_j_idx
                        rea_idx = int_i_idx
                        relation_id = 1

                objects_of_interest = 0 * np.ones((agent_num), dtype=np.int64)
                tracks_to_predict = 0 * np.ones((agent_num), dtype=np.int64)
                is_sdc = 0 * np.ones((agent_num), dtype=np.int64)
                
                objects_of_interest[inf_idx] = 1
                objects_of_interest[rea_idx] = 1
                tracks_to_predict[inf_idx] = 1
                tracks_to_predict[rea_idx] = 1
                is_sdc[inf_idx] = 1

                decoded_example_new = decoded_example.copy()
                decoded_example_new['state/is_sdc'] = is_sdc
                decoded_example_new['state/objects_of_interest'] = objects_of_interest
                decoded_example_new['state/tracks_to_predict'] = tracks_to_predict
                decoded_example_new['relation'] = [int_i_idx, int_j_idx, relation_id, agent_pair_label]
                decoded_example_new['scenario/id'] = f'{scenario_id}-{i}-{j}'
                decoded_example_new['scenario/map_id'] = hdmap_id
                
                for k, v in decoded_example_new.items():
                    if k.split('/')[0] == 'state':
                        if len(v.shape) == 2:
                            decoded_example_new[k][[inf_idx, 0], :] = v[[0, inf_idx], :]
                            if rea_idx == 0:
                                temp_reactor_idx = inf_idx
                            else:
                                temp_reactor_idx = rea_idx
                            decoded_example_new[k][[temp_reactor_idx, 1], :] = v[[1,temp_reactor_idx], :]
                        else:
                            decoded_example_new[k][[inf_idx, 0]] = v[[0, inf_idx]]
                            if rea_idx == 0:
                                temp_reactor_idx = inf_idx
                            else:
                                temp_reactor_idx = rea_idx
                            decoded_example_new[k][[temp_reactor_idx, 1]] = v[[1, temp_reactor_idx]]

                res_decoded_example = {}
                for k, v in decoded_example_new.items():
                    if 'state' in k.split('/')[0]:
                        if len(v.shape) == 2:
                            if 'traffic_light_state' in k.split('/')[0]:
                                res_decoded_example[k.split('/')[0] + '/past/'    + k.split('/')[1]] = decoded_example_new[k].copy()[0:10, :]
                                res_decoded_example[k.split('/')[0] + '/current/' + k.split('/')[1]] = decoded_example_new[k].copy()[10:11, :]
                                res_decoded_example[k.split('/')[0] + '/future/'  + k.split('/')[1]] = decoded_example_new[k].copy()[11:91, :]
                            else:
                                res_decoded_example[k.split('/')[0] + '/past/'    + k.split('/')[1]] = decoded_example_new[k].copy()[:, 0:10]
                                res_decoded_example[k.split('/')[0] + '/current/' + k.split('/')[1]] = decoded_example_new[k].copy()[:, 10:11]
                                res_decoded_example[k.split('/')[0] + '/future/'  + k.split('/')[1]] = decoded_example_new[k].copy()[:, 11:91]
                        else:
                            res_decoded_example[k] = copy.deepcopy(decoded_example_new[k])
                    else:
                        res_decoded_example[k] = copy.deepcopy(decoded_example_new[k])
                
                data_sample_list.append(res_decoded_example)

    output_path = os.path.join(output_dir, hdmap_id)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path,  scenario_id + '.pickle'), 'wb+') as f:
        pickle.dump(data_sample_list, f)

def hdmap_format_preprocess(hdmap_path, hdmap_output_dir):
    with open(hdmap_path, 'r') as f:
        hdmap_info = json.load(f)
    
    lane = 0
    lane_xyz_list = []
    lane_id = []
    lane_type = []
    
    cnt = []
    LANE = hdmap_info['LANE']

    """LANE"""
    for lanelet_id in LANE:
        centerline = LANE[lanelet_id]['centerline']
        l_type = lane_type_match[LANE[lanelet_id]['lane_type']]

        lane_xyz_list.extend(centerline)
        lane_id.extend((np.ones((len(centerline))) * lane).tolist())
        lane_type.extend((np.ones((len(centerline))) * float(l_type)).tolist())
        cnt.append(len(lane_xyz_list))

        lane = lane + 1

    lane_xyz = -1 * np.ones((len(lane_xyz_list), 3), dtype=np.float32)
    lane_xyz[:, :2] = np.array(lane_xyz_list, dtype=np.float32)
    lane_dir = np.zeros((len(lane_xyz), 3), dtype=np.float32)
    lane_type = np.array(lane_type, dtype=np.int32)
    lane_valid = np.ones((len(lane_id), 1), dtype=np.int32)
    lane_id = np.array(lane_id, dtype=np.int32)

    for i in range(len(lane_xyz)):
        if i > 0: 
            lane_dir[i - 1, 0] = (lane_xyz[i, 0] - lane_xyz[i - 1, 0]) / (np.sqrt((lane_xyz[i, 0] - lane_xyz[i - 1, 0]) ** 2 + (lane_xyz[i, 1] - lane_xyz[i - 1, 1]) **2) + 1e-9)
            lane_dir[i - 1, 1] = (lane_xyz[i, 1] - lane_xyz[i - 1, 1]) / (np.sqrt((lane_xyz[i, 0] - lane_xyz[i - 1, 0]) ** 2 + (lane_xyz[i, 1] - lane_xyz[i - 1, 1]) **2) + 1e-9)
            if i in cnt:
                lane_dir[i - 1, 0] = 0
                lane_dir[i - 1, 1] = 0

    roadgraph = {
        'roadgraph_samples/dir': lane_dir, 
        'roadgraph_samples/id': lane_id, 
        'roadgraph_samples/type': lane_type, 
        'roadgraph_samples/xyz': lane_xyz, 
        'roadgraph_samples/valid': lane_valid
    }

    hdmap_name = hdmap_path.split('/')[-1].replace('json', 'pickle')
    output_path = os.path.join(hdmap_output_dir, hdmap_name)
    with open(output_path, 'wb+') as f:
        pickle.dump(roadgraph, f)


def main():
    scenario_dir = '../int2_dataset/interaction_scenario'
    hdmap_dir = '../int2_dataset/hdmap'
    scenario_output_dir = '../int2_dataset/m2i_format/scenario'
    hdmap_output_dir = '../int2_dataset/m2i_format/hdmap'
    complete_scenario_path = os.path.join(scenario_dir, 'complete_scenario')
    scenario_dirs = [os.path.join(complete_scenario_path, f) for f in os.listdir(complete_scenario_path)]
    hdmap_path_list = [os.path.join(hdmap_dir, f) for f in os.listdir(hdmap_dir)]

    os.makedirs(hdmap_output_dir, exist_ok=True)

    # preprocess hdmap data
    for hdmap_path in hdmap_path_list:
        hdmap_format_preprocess(hdmap_path, hdmap_output_dir)

    # preprocess scenario data, need longer time
    for scenario_dir in scenario_dirs:
        complete_scenario_path_list = [os.path.join(scenario_dir, f) for f in os.listdir(scenario_dir)]
        scenario_output_dir_list = [scenario_output_dir] * len(complete_scenario_path_list)

        # scenario_format_preprocess(complete_scenario_path_list[0], scenario_output_dir_list[0])
        p_map(scenario_format_preprocess, complete_scenario_path_list[:100], scenario_output_dir_list[:100], num_cpus=0.2)

if __name__ == "__main__":
    main()