# INT2: INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import os
import numpy
import os
import numpy as np

def main():
    current_dir = os.getcwd()
    np.random.seed(42)
    scenario_floder = '../../../int2_dataset/m2i_format/scenario'
    rush_hour_txt = '../../../int2_dataset/rush_hour.txt'
    non_rush_hour_txt = '../../../int2_dataset/non_rush_hour.txt'
    domain_dir = './domain'
    split_rate = 0.7

    with open(rush_hour_txt, 'r') as f:
        rush_hour_name_list = [fp.strip().split('/')[-1] for fp in f.readlines()]
    
    with open(non_rush_hour_txt, 'r') as f:
        non_rush_hour_name_list = [fp.strip().split('/')[-1] for fp in f.readlines()]
    
    scenario_path_list = []
    scenario_dirs = [os.path.join(scenario_floder, f) for f in os.listdir(scenario_floder)]
    for scenario_dir in scenario_dirs:
        scenario_path_list.extend([os.path.join(scenario_dir, f) for f in os.listdir(scenario_dir)])

    with open(os.path.join(domain_dir, 'rush_hour_train.txt'), 'w') as f:
        f.write('')
    with open(os.path.join(domain_dir, 'non_rush_hour_train.txt'), 'w') as f:
        f.write('')
    with open(os.path.join(domain_dir, 'rush_hour_val.txt'), 'w') as f:
        f.write('')
    with open(os.path.join(domain_dir, 'non_rush_hour_val.txt'), 'w') as f:
        f.write('')

    rush_hour_train_file = open(os.path.join(domain_dir, 'rush_hour_train.txt'), 'a')
    non_rush_hour_train_file = open(os.path.join(domain_dir, 'non_rush_hour_train.txt'), 'a')
    rush_hour_val_file = open(os.path.join(domain_dir, 'rush_hour_val.txt'), 'a')
    non_rush_hour_val_file = open(os.path.join(domain_dir, 'non_rush_hour_val.txt'), 'a')

    rush_hour_path_list = []
    non_rush_hour_path_list = []
    for scenario_path in scenario_path_list:
        scenario_name = os.path.basename(scenario_path).split('.')[0]
        if scenario_name in rush_hour_name_list:
            rush_hour_path_list.append(os.path.normpath(os.path.join(current_dir, scenario_path)))
        else:
            non_rush_hour_path_list.append(os.path.normpath(os.path.join(current_dir, scenario_path)))

    np.random.shuffle(rush_hour_path_list)
    np.random.shuffle(non_rush_hour_path_list)
    

    for filepath in rush_hour_path_list[:int(len(rush_hour_path_list) * split_rate)]:
        rush_hour_train_file.write(f'{filepath}\n')

    for filepath in rush_hour_path_list[int(len(rush_hour_path_list) * split_rate):]:
        rush_hour_val_file.write(f'{filepath}\n')

    for filepath in non_rush_hour_path_list[:int(len(non_rush_hour_path_list) * split_rate)]:
        non_rush_hour_train_file.write(f'{filepath}\n')

    for filepath in non_rush_hour_path_list[int(len(non_rush_hour_path_list) * split_rate):]:
        non_rush_hour_val_file.write(f'{filepath}\n')

    rush_hour_train_file.close()
    non_rush_hour_train_file.close()
    rush_hour_val_file.close()
    non_rush_hour_val_file.close()

if __name__ == "__main__":
    main()