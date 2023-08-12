# INT2: INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

from utils.collision_utils import boundary_collision
import os
import re
import numpy as np
import argparse
from utils.vis_utils import scenario2xml
import pickle

def parse_config():
    parser = argparse.ArgumentParser(description='INT2 Dataset Visualization.')
    parser.add_argument('--scenario_path', type=str, default='int2_dataset/interaction_scenario/complete_scenario/0/010213250706-010213264206.pickle',
                         help='The scenario path to be visualized')
    parser.add_argument('--hdmap_dir', type=str, default='int2_dataset_example/hdmap',
                         help='The scenario path to be visualized')
    parser.add_argument('--output_dir', type=str, default='output/visualization', help='')
    args = parser.parse_args()

    return args

def main():
    args = parse_config()
    assert args.scenario_path != None
    hdmap_id = args.scenario_path.split('/')[-2]

    path_end_name = '/'.join(args.scenario_path.split('/')[-2:]).split('.')[0]
    xml_output_path = os.path.join(args.output_dir, path_end_name)
    hdmap_path = os.path.join(args.hdmap_dir, hdmap_id + '.json')
    os.makedirs(xml_output_path, exist_ok=True)

    interested_agents = None
    with open(args.scenario_path, 'rb+') as f:
        interaction_scenario = pickle.load(f)
    
    xml_output_path_new, MAP_RANGE, LANE, STOPLINE, CROSSWALK, JUNCTION, TRAFFIC_LIGHTS_INFO = scenario2xml(
        hdmap_path, args.scenario_path, xml_output_path)
    print(xml_output_path_new)
    ob_collision_rate, bd_collision_rate = boundary_collision(xml_output_path_new)
    print(f"ob_collision_rate: {ob_collision_rate}, bd_collision_rate: {bd_collision_rate}")

    
if __name__ == "__main__":
    main()
    
        