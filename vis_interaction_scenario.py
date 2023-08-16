# INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import warnings
import argparse
import os
import json
import pickle
warnings.filterwarnings("ignore")
from utils.vis_utils import scenario2xml, xml2video

def parse_config():
    parser = argparse.ArgumentParser(description='INT2 Dataset Visualization.')
    parser.add_argument('--scenario_path', type=str, default='int2_dataset_example/interaction_scenario/complete_scenario/8/010213355106-010213364106.pickle',
                         help='The scenario path to be visualized')
    parser.add_argument('--output_dir', type=str, default='output/visualization', help='')
    parser.add_argument('--hdmap_dir', type=str, default='int2_dataset/hdmap', help='')
    parser.add_argument('--tf_complete_id_path', type=str, default='config/tf_complete_id.json', 
                        help='The complete road ID controlled by the traffic light')
    parser.add_argument('--video_len', type=int, default=None, help='')
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

    interested_agents = None
    with open(args.scenario_path, 'rb+') as f:
        interaction_scenario = pickle.load(f)

    interested_agents = interaction_scenario['INTERACTION_INFO']['interested_agents']

    with open(args.tf_complete_id_path, 'r') as f:
        tf_complete_id_dict = json.load(f)

    tf_complete_id_map_info = tf_complete_id_dict[hdmap_id]

    xml_output_path_new, MAP_RANGE, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO = scenario2xml(
        hdmap_path, args.scenario_path, xml_output_path)

    xml2video(xml_output_path_new, video_output_path, MAP_RANGE, LANE, STOPLINE, 
              CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO, tf_complete_id_map_info, video_len=args.video_len, delete_img=False, interested_agents=interested_agents)


if __name__ == "__main__":
    main()