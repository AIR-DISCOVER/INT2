# INT2: INT2: Interactive Trajectory Prediction at Intersections
# Published at ICCV 2023
# Written by Zhijie Yan
# All Rights Reserved

import sys, os
import numpy as np
import pickle
import tensorflow as tf
import multiprocessing
import glob
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from IPython import embed