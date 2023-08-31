# INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import numpy as np
import os
import json
from math import sqrt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt
import pickle
import cv2
from utils.int2_type import tf_state_map, agent_type_map, agent_sub_type_map, lane_type_map
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from copy import deepcopy

def scenario2xml(hdmap_path, scenario_path, xml_output_path):
    with open(hdmap_path, 'r') as f:
        lanelet_baidu_data_ori = json.load(f)
    
    with open(scenario_path, 'rb+') as f:
        obstacle_tf_info = pickle.load(f)
    
    TRAFFIC_LIGHTS_INFO = obstacle_tf_info['TRAFFIC_LIGHTS_INFO']
    AGENT_INFO = obstacle_tf_info['AGENT_INFO']

    lane_type_list = ['BIKING', 'CITY_DRIVING', 'EMERGENCY_LANE','LEFT_TURN_WAITING_ZONE', 'ROUNDABOUT']
    lane_type_croad_list = ['bicycleLane', 'urban', 'driveWay', 'urban', 'urban']
    LANE = lanelet_baidu_data_ori["LANE"]
    STOPLINE = lanelet_baidu_data_ori["STOPLINE"]
    CROSSWALK = lanelet_baidu_data_ori["CROSSWALK"]
    JUNCTION = lanelet_baidu_data_ori["JUNCTION"]
    MAP_RANGE = lanelet_baidu_data_ori["MAP_RANGE"]

    d_list = [d * ' ' for d in range(2, 14, 2)]
    banchmark_id = scenario_path.split('/')[-1].split('.')[0]
    
    xml_output_path_new = os.path.join(xml_output_path, banchmark_id + '.xml')

    with open(xml_output_path_new, 'w') as data_file:
        data_file.write('')
    set_header(d_list, banchmark_id, xml_output_path_new)

    for lanelet_id in LANE:
        res = ''
        leftBound = ''
        rightBound = ''
        lane_info = LANE[lanelet_id]
        lane_type = lane_info['lane_type']
        left_boundary = lane_info['left_boundary']
        right_boundary = lane_info['right_boundary']
        predecessor = lane_info['predecessors']
        successor = lane_info['successors']
        left_neighbor_id = lane_info['left_neighbor_id']
        right_neighbor_id = lane_info['right_neighbor_id']

        res += d_list[0] + '<lanelet id="' + lanelet_id + '">' + '\n'

        leftBound +=      d_list[1] + '<leftBound>' + '\n'
        rightBound +=     d_list[1] + '<rightBound>' + '\n'

        assert len(left_boundary) == len(right_boundary)
        lane_len = len(left_boundary)
        for idx in range(lane_len):
            leftBound +=          d_list[2] + '<point>' + '\n'
            leftBound +=              d_list[3] + '<x>' + str(left_boundary[idx][0]) + "</x>" + '\n'
            leftBound +=              d_list[3] + '<y>' + str(left_boundary[idx][1]) + "</y>" + '\n'
            leftBound +=          d_list[2] + '</point>' + '\n'
            
            rightBound +=          d_list[2] + '<point>' + '\n'
            rightBound +=              d_list[3] + '<x>' + str(right_boundary[idx][0]) + "</x>" + '\n'
            rightBound +=              d_list[3] + '<y>' + str(right_boundary[idx][1]) + "</y>" + '\n'
            rightBound +=          d_list[2] + '</point>' + '\n'
        
        leftBound +=      d_list[1] + '</leftBound>' + '\n'
        rightBound +=     d_list[1] + '</rightBound>' + '\n'

        res += leftBound
        res += rightBound

        if len(predecessor) != 0:
            for i in range(len(predecessor)):
                predecessor_id = predecessor[i]
                res +=      d_list[1] + '<predecessor ref="' + f'{predecessor_id}' + '"/>' + '\n'

        if len(successor) != 0: 
            for i in range(len(successor)):
                succersor_id = successor[i]
                if succersor_id is not None:
                    res +=      d_list[1] + '<successor ref="' + f'{succersor_id}' + '"/>' + '\n'

        if len(left_neighbor_id) != 0:
            res +=      d_list[1] + '<adjacentLeft ref="' + f'{left_neighbor_id[0]}' + '" drivingDir="same"/>' + '\n'

        if len(right_neighbor_id) != 0:
            res +=      d_list[1] + '<adjacentRight ref="' + f'{right_neighbor_id[0]}' + '" drivingDir="same"/>' + '\n'

        lane_type_ = lane_type_croad_list[lane_type_list.index(lane_type_map[lane_type])]
        res +=      d_list[1] + '<laneletType>' + lane_type_ + '</laneletType>' + '\n'
        res += d_list[0] + '</lanelet>' + '\n'
        
        with open(xml_output_path_new, 'a') as data_file:
            data_file.write(res)
            
    object_id = AGENT_INFO['object_id']
    object_type = AGENT_INFO['object_type']
    object_sub_type = AGENT_INFO['object_sub_type']
    state = AGENT_INFO['state']
    
    dynamicObstacle_baidu_type_list = ['CAR', 'VAN', 'CYCLIST', 'MOTORCYCLIST', 'PEDESTRIAN', 'TRICYCLIST', 'BUS', 'TRUCK']
    dynamicObstacle_croad_type_list = ['car', 'truck', 'bicycle', 'bicycle', 'pedestrian', 'bicycle', 'bus', 'truck']
    
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
    
    acceleration = -1 * np.ones((valid.shape), dtype=np.float32)
    velocity = -1 * np.ones((valid.shape), dtype=np.float32)
    
    for i in range(valid.shape[0]):
        valid_list = valid[i].nonzero()[0]
        if len(valid_list) > 3:
            start_index = valid_list[0]
            end_index = valid_list[-1] + 1
            for j in range(start_index, end_index):
                velocity[i, j] = sqrt(velocity_x[i, j] ** 2 + velocity_y[i, j] ** 2)
                if j == start_index: 
                    velocity_first_time = sqrt(velocity_x[i, j + 1] ** 2 + velocity_y[i, j + 1] ** 2)
                    acceleration[i, j] = (velocity_first_time - velocity[i, j]) / 0.1
                elif j != end_index - 1:
                    v0 = sqrt((velocity_x[i, j - 1] ** 2 + velocity_y[i, j - 1] ** 2))
                    v2 = sqrt((velocity_x[i, j + 1] ** 2 + velocity_y[i, j + 1] ** 2))
                    acceleration[i, j] = (v2 - v0) / (0.1 * 2)
                else:
                    v_n_1 = sqrt((velocity_x[i, j - 1] ** 2 + velocity_y[i, j - 1] ** 2))
                    v_n_2 = sqrt((velocity_x[i, j - 2] ** 2 + velocity_y[i, j - 2] ** 2))
                    acceleration[i, j] = (3 * velocity[i, j] + v_n_2 - 4 * v_n_1) / (0.1 * 2)
            
    for i in range(valid.shape[0]):
        valid_list = valid[i].nonzero()[0]
        if len(valid_list) > 3:
            start_index = valid_list[0]
            end_index = valid_list[-1] + 1
            
            res = ''
            res += d_list[0] + '<dynamicObstacle id="' + '60000' + str(object_id[i]) + '">' + '\n'

            res +=      d_list[1] + '<type>' + str(dynamicObstacle_croad_type_list[dynamicObstacle_baidu_type_list.index(agent_sub_type_map[object_sub_type[i]])]) + '</type>' + '\n'
            res +=      d_list[1] + '<shape>' + '\n'
            res +=          d_list[2] + '<rectangle>' + '\n'
            res +=              d_list[3] + '<length>' + str(length[i, start_index]) + '</length>' + '\n'
            res +=              d_list[3] + '<width>' + str(width[i, start_index]) + '</width>' + '\n'
            res +=          d_list[2] + '</rectangle>' + '\n'
            res +=      d_list[1] + '</shape>' + '\n'
            res +=      d_list[1] + '<initialState>' + '\n'
            res +=          d_list[2] + '<position>' + '\n'
            res +=              d_list[3] + '<point>' + '\n'
            res +=                  d_list[4] + '<x>' + str(position_x[i, start_index]) + '</x>' + '\n'
            res +=                  d_list[4] + '<y>' + str(position_y[i, start_index]) + '</y>' + '\n'
            res +=              d_list[3] + '</point>' + '\n'
            res +=          d_list[2] + '</position>' + '\n'
            res +=          d_list[2] + '<orientation>' + '\n'
            res +=              d_list[3] + '<exact>' + str(theta[i, start_index]) + '</exact>' + '\n'
            res +=          d_list[2] + '</orientation>' + '\n'
            res +=          d_list[2] + '<time>' + '\n'
            res +=              d_list[3] + '<exact>' + str(int(start_index)) + '</exact>' + '\n'  # init state set time=0
            res +=          d_list[2] + '</time>' + '\n'
            res +=          d_list[2] + '<velocity>' + '\n'
            res +=              d_list[3] + '<exact>' + str(velocity[i, start_index]) + '</exact>' + '\n'
            res +=          d_list[2] + '</velocity>' + '\n'
            res +=          d_list[2] + '<acceleration>' + '\n'
            res +=              d_list[3] + '<exact>' + str(acceleration[i, start_index]) + '</exact>' + '\n'
            res +=          d_list[2] + '</acceleration>' + '\n'
            res +=      d_list[1] + '</initialState>' + '\n'
            res +=      d_list[1] + '<trajectory>' + '\n'
            
            
            for j in range(start_index + 1, end_index):
                res +=          d_list[2] + '<state>' + '\n'
                res +=              d_list[3] + '<position>' + '\n'
                res +=                  d_list[4] + '<point>' + '\n'
                res +=                      d_list[5] + '<x>' + str(position_x[i, j]) + '</x>' + '\n'
                res +=                      d_list[5] + '<y>' + str(position_y[i, j]) + '</y>' + '\n'
                res +=                  d_list[4] + '</point>' + '\n'
                res +=              d_list[3] + '</position>' + '\n'
                res +=              d_list[3] + '<orientation>' + '\n'
                res +=                  d_list[4] + '<exact>' + str(theta[i, j]) + '</exact>' + '\n'
                res +=              d_list[3] + '</orientation>' + '\n'
                res +=              d_list[3] + '<time>' + '\n'
                res +=                  d_list[4] + '<exact>' + str(int(j)) + '</exact>' + '\n'
                res +=              d_list[3] + '</time>' + '\n'
                res +=              d_list[3] + '<velocity>' + '\n'
                res +=                  d_list[4] + '<exact>' + str(velocity[i, j]) + '</exact>' + '\n'
                res +=              d_list[3] + '</velocity>' + '\n'
                res +=              d_list[3] + '<acceleration>' + '\n'
                res +=                  d_list[4] + '<exact>' + str(acceleration[i, j]) + '</exact>' + '\n'
                res +=              d_list[3] + '</acceleration>' + '\n'
                res +=          d_list[2] + '</state>' + '\n'
            res +=      d_list[1] + '</trajectory>' + '\n'
            res += d_list[0] + '</dynamicObstacle>' + '\n'  

            with open(xml_output_path_new, 'a') as data_file:
                data_file.write(res)
            
    set_end(xml_output_path_new)
    return xml_output_path_new, MAP_RANGE, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO

def set_header(d_list, banchmark_id, output_path):
    author = 'Apollo, Baidu Inc. and AIR, Tsinghua University'
    affiliation = 'Apollo, Baidu and AIR, Tsinghua University'
    res = ''
    res += "<?xml version='1.0' encoding='UTF-8'?>" + '\n'
    res += '<commonRoad timeStepSize="' + str(0.1) + f'" commonRoadVersion="2020a" author="{author}" affiliation="{affiliation}" source="DAIR-V2X" benchmarkID="' + banchmark_id + '" date="2022">' + '\n'
    res +=      d_list[0] + '<location>' + '\n'
    res +=          d_list[1] + '<geoNameId>999</geoNameId>' + '\n'
    res +=          d_list[1] + '<gpsLatitude>999</gpsLatitude>' + '\n'
    res +=          d_list[1] + '<gpsLongitude>999</gpsLongitude>' + '\n'
    res +=      d_list[0] + '</location>' + '\n'
    res +=      d_list[0] + '<scenarioTags>' + '\n'
    res +=          d_list[1] + '<Intersection/>' + '\n'
    res +=          d_list[1] + '<Urban/>' + '\n'
    res +=      d_list[0] + '</scenarioTags>' + '\n'

    with open(output_path, 'a') as data_file:
        data_file.write(res)

def set_end(output_path):
    res = ''
    res += '</commonRoad>'
    with open(output_path, 'a') as data_file:
            data_file.write(res)

def xml2video(xml_output_path, video_output_path, MAP_RANGE_INFO, LANE, STOPLINE, CROSSWALK, 
              JUNCTION, TRAFFIC_LIGHTS_INFO, tf_complete_id_map_info, video_len=None, delete_img=True, interested_agents=None):
    
    scenario, planning_problem_set = CommonRoadFileReader(xml_output_path).open()
    x_start, x_end = MAP_RANGE_INFO['x_start'], MAP_RANGE_INFO['x_end']
    y_start, y_end = MAP_RANGE_INFO['y_start'], MAP_RANGE_INFO['y_end']
    tf_state = TRAFFIC_LIGHTS_INFO['tf_state']
    tf_state_valid = TRAFFIC_LIGHTS_INFO['tf_state_valid']
    tf_mapping_lane_id = TRAFFIC_LIGHTS_INFO['tf_mapping_lane_id']

    tf_num = tf_state.shape[0]
    tf_num_new = tf_num
    for key, value in tf_complete_id_map_info.items():
        tf_num_new += len(value)

    tf_state_new = -1 * np.ones((tf_num_new, tf_state.shape[1]), dtype=np.int32)
    tf_state_valid_new = np.zeros((tf_num_new, tf_state.shape[1]), dtype=np.int32)
    
    tf_mapping_lane_id_new = tf_mapping_lane_id.copy()
    tf_state_new[:tf_num] = tf_state
    tf_state_valid_new[:tf_num] = tf_state_valid

    tmp_num = tf_num
    for k, v in tf_complete_id_map_info.items():
        index = tf_mapping_lane_id.index(int(k))
        for lane_id in set(v):
            tf_mapping_lane_id_new.append(lane_id)
            tf_state_new[tmp_num] = tf_state[index]
            tf_state_valid_new[tmp_num] = tf_state_valid[index]
            tmp_num += 1

    img_output_path = os.path.join(video_output_path, 'img')

    if not os.path.exists(img_output_path):
        os.makedirs(img_output_path)
    
    for path in os.listdir(img_output_path):
        file_name_ = os.path.join(img_output_path, path)
        os.remove(file_name_)

    if not video_len:
        video_len = tf_state.shape[1]
    elif video_len > tf_state.shape[1]:
        video_len = tf_state.shape[1]
    else:
        pass
    
    interested_agents_info = []
    for interested_id in interested_agents:
        dynamic_obstalce = scenario.obstacle_by_id(int('60000' + str(interested_id)))
        scenario.remove_obstacle(dynamic_obstalce)
        interested_agents_info.append(dynamic_obstalce)

    tqdm.write("Waiting for a while...")
    for i in tqdm(range(video_len)):
        plt.figure(figsize=(32, int(32 * (y_end - y_start) / (x_end - x_start))), facecolor='black')
        rnd = MPRenderer()
        scenario.draw(rnd, draw_params={'time_begin': i, 'time_end': i,"dynamic_obstacle": {"show_label": False, "draw_icon": True, "draw_shape": True, 
            'vehicle_shape': {'occupancy':{'shape': {'rectangle': {'facecolor':'white'}}}}},
            "lanelet":{
                "left_bound_color": "white",
                "right_bound_color": "white",
                "center_bound_color": "#5B5B5F",
                "unique_colors": False,
                "draw_stop_line": True,
                "stop_line_color": "#ffffff",
                "draw_line_markings": True,
                "draw_left_bound": True,
                "draw_right_bound": True,
                "draw_center_bound": True,
                "draw_border_vertices": False,
                "draw_start_and_direction": True,
                "colormap_tangent": False,
                "show_label": False,
                "draw_linewidth": 0.1,
                "facecolor": "#37373D"
            }
            })

        if len(interested_agents_info) != 0:
            for agent_idx in range(len(interested_agents_info)):
                dynamic_obstalce_now = interested_agents_info[agent_idx]
                dynamic_obstalce_now.draw(rnd, draw_params={'time_begin': i, 'time_end': i,"dynamic_obstacle": {"show_label": False, "draw_icon": True, "draw_shape": True, 
                    'vehicle_shape': {'occupancy':{'shape': {'rectangle': {'facecolor':'orange'}}}}}})
        
        rnd.render()

        if LANE:
            for lanelet_id, lane in LANE.items():
                lane_centerline = np.array(lane['centerline'])
                lane_left_boundary = np.array(lane['left_boundary'])
                lane_right_boundary = np.array(lane['right_boundary'])
                
                if int(lanelet_id) in tf_mapping_lane_id:
                    index = tf_mapping_lane_id.index(int(lanelet_id))
                    centerline = np.array(lane['centerline'])
                    left_boundary = np.array(lane['left_boundary'])
                    right_boundary = np.array(lane['right_boundary'])
                    c_position_x = centerline[:, 0]
                    c_position_y = centerline[:, 1]
                    l_position_x = left_boundary[:, 0]
                    l_position_y = left_boundary[:, 1]
                    r_position_x = right_boundary[:, 0]
                    r_position_y = right_boundary[:, 1]
                    marker = 'o'
                    if tf_state_valid[index, i]:
                        color = tf_state_map[tf_state[index, i]]
                    else:
                        color = 'yellow'

                    # plt.plot(lane_left_boundary[:, 0], lane_left_boundary[:, 1], color=color, linewidth=2, zorder=30)
                    # plt.plot(lane_right_boundary[:, 0], lane_right_boundary[:, 1], color=color, linewidth=2, zorder=30)
                    
                    lane_width = min(np.sqrt((l_position_x - r_position_x) ** 2 + (l_position_y - r_position_y) ** 2))
                    s = 100
                    if lane_width <= 2.5:
                        s = 60
                    plt.scatter([c_position_x[0], l_position_x[0], r_position_x[0]], [c_position_y[0], l_position_y[0], r_position_y[0]], color=color, s=s, zorder=31, alpha=0.7, marker=marker)
                    plt.plot([l_position_x[0], r_position_x[0]], [l_position_y[0], r_position_y[0]], color='black', linewidth=15, zorder=30, alpha=0.6)
                
                
                if int(lanelet_id) in tf_mapping_lane_id_new:
                    index = tf_mapping_lane_id_new.index(int(lanelet_id))
                    if tf_state_valid_new[index, i]:
                        if tf_state_map[tf_state_new[index, i]] == 'GREEN':
                            zorder_num = 16
                            alpha = 0.8
                            linewidth_num = 1
                            plt.fill(list(lane_left_boundary[:, 0]) + list(lane_right_boundary[:, 0][::-1]), \
                                    list(lane_left_boundary[:, 1]) + list(lane_right_boundary[:, 1][::-1]), color=tf_state_map[tf_state_new[index, i]], zorder=zorder_num, alpha=0.2)
                        elif tf_state_map[tf_state_new[index, i]] == 'RED':
                            zorder_num = 15
                            alpha = 0.5
                            linewidth_num = 0.5
                        plt.plot(lane_left_boundary[:, 0], lane_left_boundary[:, 1], color=tf_state_map[tf_state_new[index, i]], linewidth=linewidth_num, zorder=zorder_num, alpha=alpha)
                        plt.plot(lane_right_boundary[:, 0], lane_right_boundary[:, 1], color=tf_state_map[tf_state_new[index, i]], linewidth=linewidth_num, zorder=zorder_num, alpha=alpha)
                            

        if STOPLINE:
            for stopline_id in STOPLINE.keys():
                centerline = np.array(STOPLINE[stopline_id]['centerline'])
                x = list(centerline[:, 0])
                y = list(centerline[:, 1])
                x.append(x[0])
                y.append(y[0])
                plt.plot(x, y, color='red', linewidth=2, zorder=15, alpha=0.8)

        if CROSSWALK:
            for crosswalk_id in CROSSWALK.keys():
                polygon = np.array(CROSSWALK[crosswalk_id]['polygon'])
                x = list(polygon[:, 0])
                y = list(polygon[:, 1])
                x.append(x[0])
                y.append(y[0])
                plt.plot(x, y, color='#FBFF34', linewidth=2, zorder=15, alpha=0.8)

        if JUNCTION:
            for junction_id in JUNCTION.keys():
                polygon = np.array(JUNCTION[junction_id]['polygon'])
                x = list(polygon[:, 0])
                y = list(polygon[:, 1])
                x.append(x[0])
                y.append(y[0])
                plt.plot(x, y, color='#007FD4', linewidth=2, zorder=15, alpha=0.8)

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        
        plt.savefig(os.path.join(img_output_path, f'frame{i}' + '.png'), bbox_inches='tight')
        plt.close()
        
    image_ori = cv2.imread(img_output_path + '/frame0.png')
    video_size = (image_ori.shape[1], image_ori.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_name = 'video.mp4'

    video = cv2.VideoWriter(os.path.join(video_output_path, video_name), fourcc, 10, video_size, True)
    list_file = os.listdir(img_output_path)
    list_file.sort(key=lambda x:int(float(x[5:-4])))

    for i, file in enumerate(list_file):
        filename = os.path.join(img_output_path, file)
        frame = cv2.imread(filename)
        
        video.write(frame)
    video.release()
    
    if delete_img:
        for path in os.listdir(img_output_path):
            file_name_ = os.path.join(img_output_path, path)
            os.remove(file_name_)
    

def xml2video_split(xml_output_path, video_output_path, MAP_RANGE_INFO, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO, 
                    tf_complete_id_map_info, video_len=None, delete_img=True, split_interaction_info=None):
    
    scenario, planning_problem_set = CommonRoadFileReader(xml_output_path).open()
    x_start, x_end = MAP_RANGE_INFO['x_start'], MAP_RANGE_INFO['x_end']
    y_start, y_end = MAP_RANGE_INFO['y_start'], MAP_RANGE_INFO['y_end']
    tf_state = TRAFFIC_LIGHTS_INFO['tf_state']
    tf_state_valid = TRAFFIC_LIGHTS_INFO['tf_state_valid']
    tf_mapping_lane_id = TRAFFIC_LIGHTS_INFO['tf_mapping_lane_id']

    tf_num = tf_state.shape[0]
    tf_num_new = tf_num
    for key, value in tf_complete_id_map_info.items():
        tf_num_new += len(value)

    tf_state_new = -1 * np.ones((tf_num_new, tf_state.shape[1]), dtype=np.int32)
    tf_state_valid_new = np.zeros((tf_num_new, tf_state.shape[1]), dtype=np.int32)
    
    tf_mapping_lane_id_new = tf_mapping_lane_id.copy()
    tf_state_new[:tf_num] = tf_state
    tf_state_valid_new[:tf_num] = tf_state_valid

    tmp_num = tf_num
    for k, v in tf_complete_id_map_info.items():
        index = tf_mapping_lane_id.index(int(k))
        for lane_id in set(v):
            tf_mapping_lane_id_new.append(lane_id)
            tf_state_new[tmp_num] = tf_state[index]
            tf_state_valid_new[tmp_num] = tf_state_valid[index]
            tmp_num += 1
    
    if split_interaction_info:
        for key, value in split_interaction_info.items():
            interested_agents = value['interested_agents']
            split_time_91 = value['split_time_91']

            img_output_path = os.path.join(video_output_path, f'img{key}')

            os.makedirs(img_output_path, exist_ok=True)
            
            for path in os.listdir(img_output_path):
                file_name_ = os.path.join(img_output_path, path)
                os.remove(file_name_)

            interested_agents_info = []
            for interested_id in interested_agents:
                scenario_new = deepcopy(scenario)
                dynamic_obstalce = scenario_new.obstacle_by_id(int('60000' + str(interested_id)))
                scenario_new.remove_obstacle(dynamic_obstalce)
                interested_agents_info.append(dynamic_obstalce)
                
            tqdm.write("Waiting for a while...")
            for i in tqdm(range(split_time_91[0], split_time_91[1] + 1, 1)):
                plt.figure(figsize=(32, int(32 * (y_end - y_start) / (x_end - x_start))), facecolor='#1E1E1E')
                rnd = MPRenderer()
                scenario_new.draw(rnd, draw_params={'time_begin': i, 'time_end': i,"dynamic_obstacle": {"show_label": False, "draw_icon": True, "draw_shape": True, 
                    'vehicle_shape': {'occupancy':{'shape': {'rectangle': {'facecolor':'white'}}}}},
                    "lanelet":{
                        "left_bound_color": "white",
                        "right_bound_color": "white",
                        "center_bound_color": "#5B5B5F",
                        "unique_colors": False,
                        "draw_stop_line": True,
                        "stop_line_color": "#ffffff",
                        "draw_line_markings": True,
                        "draw_left_bound": True,
                        "draw_right_bound": True,
                        "draw_center_bound": True,
                        "draw_border_vertices": False,
                        "draw_start_and_direction": True,
                        "colormap_tangent": False,
                        "show_label": False,
                        "draw_linewidth": 0.1,
                        "facecolor": "#37373D"
                    }
                    })

                if len(interested_agents_info) != 0:
                    for agent_idx in range(len(interested_agents_info)):
                        dynamic_obstalce_now = interested_agents_info[agent_idx]
                        dynamic_obstalce_now.draw(rnd, draw_params={'time_begin': i, 'time_end': i,"dynamic_obstacle": {"show_label": False, "draw_icon": True, "draw_shape": True, 
                            'vehicle_shape': {'occupancy':{'shape': {'rectangle': {'facecolor':'orange'}}}}}})
                
                rnd.render()

                if LANE:
                    for lanelet_id, lane in LANE.items():
                        lane_centerline = np.array(lane['centerline'])
                        lane_left_boundary = np.array(lane['left_boundary'])
                        lane_right_boundary = np.array(lane['right_boundary'])
                        
                        if int(lanelet_id) in tf_mapping_lane_id:
                            index = tf_mapping_lane_id.index(int(lanelet_id))
                            centerline = np.array(lane['centerline'])
                            left_boundary = np.array(lane['left_boundary'])
                            right_boundary = np.array(lane['right_boundary'])
                            c_position_x = centerline[:, 0]
                            c_position_y = centerline[:, 1]
                            l_position_x = left_boundary[:, 0]
                            l_position_y = left_boundary[:, 1]
                            r_position_x = right_boundary[:, 0]
                            r_position_y = right_boundary[:, 1]
                            marker = 'o'
                            if tf_state_valid[index, i]:
                                color = tf_state_map[tf_state[index, i]]
                            else:
                                color = 'yellow'

                            lane_width = min(np.sqrt((l_position_x - r_position_x) ** 2 + (l_position_y - r_position_y) ** 2))
                            s = 100
                            if lane_width <= 2.5:
                                s = 60
                            plt.scatter([c_position_x[0], l_position_x[0], r_position_x[0]], [c_position_y[0], l_position_y[0], r_position_y[0]], color=color, s=s, zorder=31, alpha=0.7, marker=marker)
                            plt.plot([l_position_x[0], r_position_x[0]], [l_position_y[0], r_position_y[0]], color='black', linewidth=15, zorder=30, alpha=0.6)
                        
                        
                        if int(lanelet_id) in tf_mapping_lane_id_new:
                            index = tf_mapping_lane_id_new.index(int(lanelet_id))
                            if tf_state_valid_new[index, i]:
                                if tf_state_map[tf_state_new[index, i]] == 'GREEN':
                                    zorder_num = 16
                                    alpha = 0.8
                                    linewidth_num = 1
                                    plt.fill(list(lane_left_boundary[:, 0]) + list(lane_right_boundary[:, 0][::-1]), \
                                            list(lane_left_boundary[:, 1]) + list(lane_right_boundary[:, 1][::-1]), color=tf_state_map[tf_state_new[index, i]], zorder=zorder_num, alpha=0.2)
                                elif tf_state_map[tf_state_new[index, i]] == 'RED':
                                    zorder_num = 15
                                    alpha = 0.5
                                    linewidth_num = 0.5
                                plt.plot(lane_left_boundary[:, 0], lane_left_boundary[:, 1], color=tf_state_map[tf_state_new[index, i]], linewidth=linewidth_num, zorder=zorder_num, alpha=alpha)
                                plt.plot(lane_right_boundary[:, 0], lane_right_boundary[:, 1], color=tf_state_map[tf_state_new[index, i]], linewidth=linewidth_num, zorder=zorder_num, alpha=alpha)

                if STOPLINE:
                    for stopline_id in STOPLINE.keys():
                        centerline = np.array(STOPLINE[stopline_id]['centerline'])
                        x = list(centerline[:, 0])
                        y = list(centerline[:, 1])
                        x.append(x[0])
                        y.append(y[0])
                        plt.plot(x, y, color='red', linewidth=2, zorder=15, alpha=0.8)

                if CROSSWALK:
                    for crosswalk_id in CROSSWALK.keys():
                        polygon = np.array(CROSSWALK[crosswalk_id]['polygon'])
                        x = list(polygon[:, 0])
                        y = list(polygon[:, 1])
                        x.append(x[0])
                        y.append(y[0])
                        plt.plot(x, y, color='#FBFF34', linewidth=2, zorder=15, alpha=0.8)

                if JUNCTION:
                    for junction_id in JUNCTION.keys():
                        polygon = np.array(JUNCTION[junction_id]['polygon'])
                        x = list(polygon[:, 0])
                        y = list(polygon[:, 1])
                        x.append(x[0])
                        y.append(y[0])
                        plt.plot(x, y, color='#007FD4', linewidth=2, zorder=15, alpha=0.8)

                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                
                plt.savefig(os.path.join(img_output_path, f'frame{i}' + '.png'), bbox_inches='tight')
                plt.close()
            
            list_file = os.listdir(img_output_path)
            list_file.sort(key=lambda x:int(float(x[5:-4])))

            image_ori = cv2.imread(os.path.join(img_output_path, list_file[0]))
            video_size = (image_ori.shape[1], image_ori.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            video_name = f'video{key}.mp4'

            video = cv2.VideoWriter(os.path.join(video_output_path, video_name), fourcc, 10, video_size, True)
            

            for i, file in enumerate(list_file):
                filename = os.path.join(img_output_path, file)
                frame = cv2.imread(filename)
                
                video.write(frame)
            video.release()
            
            if delete_img:
                for path in os.listdir(img_output_path):
                    file_name_ = os.path.join(img_output_path, path)
                    os.remove(file_name_)
    else:
        return

    

def main():
    scenario_path = '../int_dataset/scenario/1/050213405400-050213432800.pickle'
    output_dir = '../output/visualization'
    tf_complete_id_path = '../config/tf_complete_id.json'
    hdmap_dir = '../int_dataset/hdmap'

    path_end_name = '/'.join(scenario_path.split('/')[-2:])
    xml_output_path = os.path.join(output_dir, path_end_name)
    video_output_path = os.path.join(output_dir, path_end_name)
    hdmap_id = scenario_path.split('/')[-2]
    hdmap_path = os.path.join(hdmap_dir, hdmap_id + '.json')

    os.makedirs(xml_output_path, exist_ok=True)
    os.makedirs(xml_output_path, exist_ok=True)

    with open(tf_complete_id_path, 'r') as f:
        tf_complete_id_dict = json.load(f)

    tf_complete_id_map_info = tf_complete_id_dict[hdmap_id]

    xml_output_path_new, MAP_RANGE, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO = scenario2xml(hdmap_path, scenario_path, xml_output_path)

    xml2video(xml_output_path_new, video_output_path, MAP_RANGE, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO, tf_complete_id_map_info, video_len=10)

if __name__ == "__main__":
    main()