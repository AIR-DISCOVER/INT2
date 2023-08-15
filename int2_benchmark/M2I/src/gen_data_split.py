import os
import numpy as np

from IPython import embed

root_path = '/DATA/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle/peak'
save_path = '/DATA_EDS/yanzj/workspace/code/INT2/dataset/INT2_merge/vehicle-vehicle/peak/peak.txt'

all_path = []

map_list = sorted([f for f in os.listdir(root_path)])
for map_item in map_list:
    map_item_path = os.path.join(root_path, map_item)
    if not os.path.isdir(map_item_path):
        continue
    scene_list = sorted([os.path.join(root_path, map_item, f) for f in os.listdir(map_item_path)], key=lambda x: x[-12:-7])

    all_path.extend(scene_list)

np.savetxt(save_path, all_path, fmt='%s')
