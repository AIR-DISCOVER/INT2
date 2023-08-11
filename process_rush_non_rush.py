import os
import numpy
from IPython import embed

def main():
    rush_hour_path = 'int2_dataset/rush_hour.txt'
    non_rush_hour_path = 'int2_dataset/rush_hour.txt'
    with open(rush_hour_path, 'r') as f:
        rush_hour_path_list = f.readlines()
    
    rush_hour_path_list = [f.strip() for f in rush_hour_path_list]
    rush_hour_path_list = sorted(rush_hour_path_list, key=lambda x: int(x.split('/')[-2]))

    with open(rush_hour_path, 'w') as f:
        for path in rush_hour_path_list:
            f.write(f'{path}\n')

if __name__ == "__main__":
    main()