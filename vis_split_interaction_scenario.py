# INT2: INT2: Interactive Trajectory Prediction at Intersections
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
from IPython import embed
import pickle
import cv2
import warnings
import argparse
warnings.filterwarnings("ignore")
from tqdm import tqdm
from utils.vis_utils import scenario2xml, xml2video_split

def parse_config():
    parser = argparse.ArgumentParser(description='INT2 Dataset Split Data Visualization.')
    parser.add_argument('--scenario_path', '--s', type=str, default='int2_dataset_example/interaction_scenario/complete_scenario/8/010213355106-010213364106.pickle',
                         help='The scenario path to be visualized')
    parser.add_argument('--output_dir', type=str, default='output/visualization', help='')
    parser.add_argument('--hdmap_dir', type=str, default='int2_dataset/hdmap', help='')
    parser.add_argument('--tf_complete_id_path', type=str, default='config/tf_complete_id.json', 
                        help='The complete road ID controlled by the traffic light')
    parser.add_argument('--video_len', type=int, default=None, help='')
    parser.add_argument('--split_interaction_scenario_dir', type=str, default='int2_dataset_example/interaction_scenario/split_scenario', 
                        help='The folder where the complete scene interaction information is located')
    args = parser.parse_args()

    return args

def main():
    args = parse_config()
    assert args.scenario_path != None
    assert args.tf_complete_id_path != None

    hdmap_id = args.scenario_path.split('/')[-2]

    path_end_name = '/'.join(args.scenario_path.split('/')[-2:]).split('.')[0]

    xml_output_path = os.path.join(args.output_dir, path_end_name)
    video_output_path = os.path.join(args.output_dir, path_end_name)
    hdmap_path = os.path.join(args.hdmap_dir, hdmap_id + '.json')
    
    os.makedirs(xml_output_path, exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    with open(args.tf_complete_id_path, 'r') as f:
        tf_complete_id_dict = json.load(f)

    tf_complete_id_map_info = tf_complete_id_dict[hdmap_id]

    if args.split_interaction_scenario_dir != None:
        split_interaction_info_file = os.path.join(args.split_interaction_scenario_dir, '/'.join(args.scenario_path.split('/')[-2:]))
        with open(split_interaction_info_file, 'rb+') as f:
            split_interaction_info = pickle.load(f)
            
    xml_output_path_new, MAP_RANGE, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO = scenario2xml(
        hdmap_path, args.scenario_path, xml_output_path)

    xml2video_split(xml_output_path_new, video_output_path, MAP_RANGE, LANE, STOPLINE, 
            CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO, tf_complete_id_map_info, video_len=args.video_len, delete_img=False, split_interaction_info=split_interaction_info)


def multi_process():
    split_scenario_floder = 'int2_dataset/interaction_scenario/split_scenario'
    output_dir = 'output/visualization'
    hdmap_dir = 'int2_dataset/hdmap'
    tf_complete_id_path = 'config/tf_complete_id.json'

    scenario_dirs = [os.path.join(split_scenario_floder, f) for f in os.listdir(split_scenario_floder)]
    with open(tf_complete_id_path, 'r') as f:
        tf_complete_id_dict = json.load(f)
    
    for s_dir in scenario_dirs:
        if s_dir.split('/')[-1] != '7':
            continue
        file_paths = [os.path.join(s_dir, f) for f in os.listdir(s_dir)]
        for idx, file_path in enumerate(file_paths):
            if idx > 0:
                continue
            complete_path = file_path.replace('split', 'complete')
            hdmap_id = complete_path.split('/')[-2]
            path_end_name = '/'.join(complete_path.split('/')[-2:]).split('.')[0]
            xml_output_path = os.path.join(output_dir, path_end_name)
            video_output_path = os.path.join(output_dir, path_end_name)
            hdmap_path = os.path.join(hdmap_dir, hdmap_id + '.json')
            os.makedirs(xml_output_path, exist_ok=True)
            os.makedirs(video_output_path, exist_ok=True)
            tf_complete_id_map_info = tf_complete_id_dict[hdmap_id]
            with open(file_path, 'rb+') as f:
                split_interaction_info = pickle.load(f)

            xml_output_path_new, MAP_RANGE, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO = scenario2xml(
                hdmap_path, complete_path, xml_output_path)
            
            xml2video_split(xml_output_path_new, video_output_path, MAP_RANGE, LANE, STOPLINE, 
                CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO, tf_complete_id_map_info, video_len=None, delete_img=False, split_interaction_info=split_interaction_info)

if __name__ == "__main__":
    main()
    # multi_process()