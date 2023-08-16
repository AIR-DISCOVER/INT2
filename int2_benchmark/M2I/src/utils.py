import argparse
import inspect
import json
import math
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
import time
import pdb
from collections import defaultdict
from multiprocessing import Process
from random import randint
from typing import Dict, List, Tuple, NamedTuple, Any, Union, Optional
import svgpath2mpl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.path import Path
from matplotlib.pyplot import MultipleLocator
from torch import Tensor
from enum import IntEnum
from . import globals, utils_cython, structs
_False = False
if _False:
    import utils_cython


def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    # Required parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("-e", "--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true')
    # parser.add_argument("--data_dir",
    #                     nargs='*',
    #                     default=["train/data/"],
    #                     type=str)
    parser.add_argument("--data_txt",
                        required=True,
                        type=str)
    parser.add_argument("--hdmap_dir",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir", default="tmp/", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--temp_file_dir", default=None, type=str)
    parser.add_argument("--train_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="Path for loading a pre-trained model for fine-tune or just validation predicts")
    parser.add_argument("--validation_model",
                        default=21,
                        type=int,
                        help="Pass a number to load model of a specified epoch from current output directory")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.3,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=30.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--hidden_size",
                        default=64,
                        type=int)
    parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--sub_graph_depth",
                        default=3,
                        type=int)
    parser.add_argument("--global_graph_depth",
                        default=1,
                        type=int)
    parser.add_argument("--debug",
                        action='store_true')
    parser.add_argument("--initializer_range",
                        default=0.02,
                        type=float)
    parser.add_argument("--sub_graph_batch_size",
                        default=8000,
                        type=int)
    parser.add_argument("-d", "--distributed_training",
                        nargs='?',
                        # default=8,
                        default=0,
                        const=4,
                        type=int)
    parser.add_argument("--cuda_visible_device_num",
                        default=None,
                        type=int)
    parser.add_argument("--use_map",
                        action='store_true')
    parser.add_argument("--reuse_temp_file",
                        action='store_true')
    parser.add_argument("--old_version",
                        action='store_true')
    parser.add_argument("--max_distance",
                        default=50.0,
                        type=float)
    parser.add_argument("--no_sub_graph",
                        action='store_true')
    parser.add_argument("--no_agents",
                        action='store_true')
    parser.add_argument("--other_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-ep", "--eval_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-tp", "--train_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("--not_use_api",
                        action='store_true')
    parser.add_argument("--core_num",
                        default=1,
                        type=int)
    parser.add_argument("--visualize",
                        action='store_true')
    parser.add_argument("--train_extra",
                        action='store_true')
    parser.add_argument("--use_centerline",
                        action='store_true')
    parser.add_argument("--autoregression",
                        nargs='?',
                        default=None,
                        const=2,
                        type=int)
    parser.add_argument("--lstm",
                        action='store_true')
    parser.add_argument("--add_prefix",
                        default=None)
    parser.add_argument("--attention_decay",
                        action='store_true')
    parser.add_argument("--placeholder",
                        default=0.0,
                        type=float)
    parser.add_argument("--multi",
                        nargs='?',
                        default=None,
                        const=6,
                        type=int)
    parser.add_argument("--method_span",
                        nargs='*',
                        default=[0, 1],
                        type=int)
    parser.add_argument("--nms_threshold",
                        default=None,
                        type=float)
    parser.add_argument("--stage_one_K", type=int)
    parser.add_argument("--master_port", default='12355')
    parser.add_argument("--gpu_split",
                        nargs='?',
                        default=0,
                        const=2,
                        type=int)
    parser.add_argument("--waymo",
                        action='store_true')
    parser.add_argument("--argoverse",
                        action='store_true')
    parser.add_argument("--nuscenes",
                        action='store_true')
    parser.add_argument("--future_frame_num",
                        default=80,
                        type=int)
    parser.add_argument("--future_test_frame_num",
                        default=16,
                        type=int)
    parser.add_argument("--single_agent",
                        action='store_true',
                        default=True)
    parser.add_argument("--agent_type",
                        default=None,
                        type=str)
    parser.add_argument("--inter_agent_types",
                        default=None,
                        nargs=2,
                        type=str)
    parser.add_argument("--mode_num",
                        default=6,
                        type=int)
    parser.add_argument("--joint_target_each",
                        default=80,
                        type=int)
    parser.add_argument("--joint_target_type",
                        type=str,
                        choices=["no", "single", "pair"],
                        default="no",
                        help="Options to get joint target.")
    parser.add_argument("--joint_nms_type",
                        type=str,
                        choices=["or", "and"],
                        default="and",
                        help="Options to select joint goals using NMS.")
    parser.add_argument("--debug_mode",
                        action='store_true')
    parser.add_argument("--traj_loss_coeff",
                        default=1.0,
                        type=float,
                        help="Coefficient of trajectory loss.")
    parser.add_argument("--short_term_loss_coeff",
                        default=0.0,
                        type=float,
                        help="Coefficient of loss at 3s and 5s.")
    parser.add_argument("--classify_sub_goals",
                        action='store_true',
                        help="Classify goals at 3s and 5s.")
    parser.add_argument("--config",
                        default=None,
                        type=str,
                        help="Name of config file.")
    # the following params are used for reactor predictions following I->R
    # parser.add_argument("--relation_file_path",
    #                     default=None,
    #                     type=str,
    #                     help="Path of interaction relation file.")
    parser.add_argument("--influencer_pred_file_path",
                        default=None,
                        type=str,
                        help="Path of the prediction result of the influencers predictions.")
    parser.add_argument("--relation_pred_file_path",
                        default=None,
                        type=str,
                        help="Path of the prediction result of the relationship.")
    parser.add_argument("--reverse_pred_relation",
                        action='store_true',
                        default=False)
    # for saving prediction results when eval
    parser.add_argument("--eval_rst_saving_number",
                        type=str,
                        default=None,
                        help="Path of the file to save 6 prediction results.")
    parser.add_argument("--eval_exp_path",
                        type=str,
                        default=None,
                        help="Path for saving eval result")
    parser.add_argument("--infMLP",
                        default=0,
                        type=int)
    parser.add_argument("--relation_pred_threshold",
                        default=0.8,
                        type=float)
    parser.add_argument("--direct_relation_path",
                        default=None,
                        type=str,
                        help="Path of the prediction result of the relationship.")
    parser.add_argument("--all_agent_ids_path",
                        default=None,
                        type=str,
                        help="Path of the ids of all relevant agents for each scenario.")
    parser.add_argument("--vehicle_r_pred_threshold",
                        default=None,
                        type=float)


class Args:
    # data_dir = None
    data_kind = None
    debug = None
    train_batch_size = None
    seed = None
    eval_batch_size = None
    distributed_training = None
    cuda_visible_device_num = None
    log_dir = None
    learning_rate = None
    do_eval = None
    hidden_size = None
    sub_graph_depth = None
    global_graph_depth = None
    train_batch_size = None
    num_train_epochs = None
    initializer_range = None
    sub_graph_batch_size = None
    temp_file_dir = None
    output_dir = None
    use_map = None
    reuse_temp_file = None
    old_version = None
    model_recover_path = None
    do_train = None
    max_distance = None
    no_sub_graph = None
    other_params: Dict = None
    eval_params = None
    train_params = None
    no_agents = None
    not_use_api = None
    core_num = None
    visualize = None
    train_extra = None
    hidden_dropout_prob = None
    use_centerline = None
    autoregression = None
    lstm = None
    add_prefix = None
    attention_decay = None
    do_test = None
    placeholder = None
    multi = None
    method_span = None
    waymo = None
    argoverse = None
    nuscenes = None
    single_agent = None
    agent_type = None
    future_frame_num = None
    no_cuda = None
    mode_num = None
    nms_threshold = None
    inter_agent_types = None
    config = None

class Normalizer:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.origin = rotate(0.0 - x, 0.0 - y, yaw)

    def __call__(self, points, reverse=False):
        points = np.array(points)
        if points.shape == (2,):
            points.shape = (1, 2)
        assert len(points.shape) <= 3
        if len(points.shape) == 3:
            for each in points:
                each[:] = self.__call__(each, reverse)
        else:
            assert len(points.shape) == 2
            for point in points:
                if reverse:
                    point[0], point[1] = rotate(point[0] - self.origin[0],
                                                point[1] - self.origin[1], -self.yaw)
                else:
                    point[0], point[1] = rotate(point[0] - self.x,
                                                point[1] - self.y, self.yaw)

        return points


class CustomMarker(Path):
    def __init__(self, icon, az):
        # if icon == "icon":
        #     verts = iconMat
        # svg = """<svg t="1624195118046" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="19465" xmlns:xlink="http://www.w3.org/1999/xlink" width="700" height="700"><defs><style type="text/css"></style></defs><path d="M812.875093 411.578027l-0.003413 0.01536-43.562667-11.671894V203.436373c0-102.367573-112.216747-185.35424-250.63936-185.35424s-250.641067 82.986667-250.641066 185.35424l-0.360107 10.238294v187.89376l-41.89696 11.226453-0.006827-0.01536c-26.519893 7.120213-44.946773 24.33536-41.166506 38.469973l47.930026-12.84096 0.003414 0.013654 35.136853-9.413974v484.061867l0.360107 7.022933c0 48.899413 112.218453 88.546987 250.641066 88.546987s250.63936-39.645867 250.63936-88.546987V427.36128l36.800854 9.86112 0.00512-0.01536 47.92832 12.84096c3.77856-14.134613-14.64832-31.34976-41.168214-38.469973zM658.152107 87.01952c13.34272-9.344 37.459627 2.075307 53.86752 25.506133s18.889387 49.998507 5.543253 59.342507c-13.34272 9.347413-37.46304-2.0736-53.86752-25.50272-16.406187-23.432533-18.891093-50.00192-5.543253-59.34592z m65.14176 231.66976l-42.922667 82.507093c-88.410453-28.182187-231.00416-29.134507-323.060053-2.84672l-41.96352-79.786666c92.73856-87.42912 315.33056-87.386453 407.94624 0.126293zM325.08416 111.418027c16.406187-23.430827 40.521387-34.850133 53.865813-25.506134 13.346133 9.344 10.862933 35.91168-5.543253 59.342507-16.402773 23.430827-40.521387 34.850133-53.865813 25.504427-13.34784-9.340587-10.86464-35.909973 5.543253-59.3408zM307.2 348.16c28.352853 17.481387 41.51808 150.084267 38.674773 276.48H307.2V348.16z m0 501.76V648.533333h37.94432c-3.43552 88.183467-14.849707 169.470293-34.530987 201.386667h-3.413333z m15.423147 21.143893l47.071573-118.454613 0.116053-0.114347c32.37888 32.37888 262.442667 30.34112 295.401814-2.618026l1.11104 0.269653 49.6896 117.439147c-43.892053 43.88864-350.266027 46.600533-393.39008 3.478186zM737.08032 846.506667h-3.413333c-19.679573-31.916373-31.095467-113.2032-34.52928-201.386667h37.942613v201.386667z m0-225.28h-38.673067c-2.843307-126.395733 10.320213-258.998613 38.673067-276.48v276.48z" fill="#1296db" p-id="19466"></path></svg>"""
        svg = "M812.875093 411.578027l-0.003413 0.01536-43.562667-11.671894V203.436373c0-102.367573-112.216747-185.35424-250.63936-185.35424s-250.641067 82.986667-250.641066 185.35424l-0.360107 10.238294v187.89376l-41.89696 11.226453-0.006827-0.01536c-26.519893 7.120213-44.946773 24.33536-41.166506 38.469973l47.930026-12.84096 0.003414 0.013654 35.136853-9.413974v484.061867l0.360107 7.022933c0 48.899413 112.218453 88.546987 250.641066 88.546987s250.63936-39.645867 250.63936-88.546987V427.36128l36.800854 9.86112 0.00512-0.01536 47.92832 12.84096c3.77856-14.134613-14.64832-31.34976-41.168214-38.469973zM658.152107 87.01952c13.34272-9.344 37.459627 2.075307 53.86752 25.506133s18.889387 49.998507 5.543253 59.342507c-13.34272 9.347413-37.46304-2.0736-53.86752-25.50272-16.406187-23.432533-18.891093-50.00192-5.543253-59.34592z m65.14176 231.66976l-42.922667 82.507093c-88.410453-28.182187-231.00416-29.134507-323.060053-2.84672l-41.96352-79.786666c92.73856-87.42912 315.33056-87.386453 407.94624 0.126293zM325.08416 111.418027c16.406187-23.430827 40.521387-34.850133 53.865813-25.506134 13.346133 9.344 10.862933 35.91168-5.543253 59.342507-16.402773 23.430827-40.521387 34.850133-53.865813 25.504427-13.34784-9.340587-10.86464-35.909973 5.543253-59.3408zM307.2 348.16c28.352853 17.481387 41.51808 150.084267 38.674773 276.48H307.2V348.16z m0 501.76V648.533333h37.94432c-3.43552 88.183467-14.849707 169.470293-34.530987 201.386667h-3.413333z m15.423147 21.143893l47.071573-118.454613 0.116053-0.114347c32.37888 32.37888 262.442667 30.34112 295.401814-2.618026l1.11104 0.269653 49.6896 117.439147c-43.892053 43.88864-350.266027 46.600533-393.39008 3.478186zM737.08032 846.506667h-3.413333c-19.679573-31.916373-31.095467-113.2032-34.52928-201.386667h37.942613v201.386667z m0-225.28h-38.673067c-2.843307-126.395733 10.320213-258.998613 38.673067-276.48v276.48z"
        # import xml.etree.ElementTree as etree
        # from six import StringIO
        # tree = etree.parse(StringIO(svg))
        # root = tree.getroot()
        az = az + math.radians(180)
        verts = svgpath2mpl.parse_path(svg).vertices
        verts[:, 0] -= (867 - 180) / 2 + 180
        verts[:, 1] -= (1008 - 18) / 2 + 18
        vertices = rot(verts, az)
        super().__init__(vertices, codes=svgpath2mpl.parse_path(svg).codes)


class Pool:
    def __init__(self, core_num, files, run):
        self.core_num = core_num
        self.queue = multiprocessing.Queue(core_num)
        self.result_queue = multiprocessing.Queue(core_num)
        self.processes = [multiprocessing.Process(target=pool_forward, args=(rank, self.queue, self.result_queue, run,)) for rank in
                          range(self.core_num)]
        self.files = files
        for each in self.processes:
            each.start()
        for file in files:
            assert file is not None
            self.queue.put(file)

    def join(self):
        results = []
        for i in range(len(self.files)):
            results.append(self.result_queue.get())

        while not self.queue.empty():
            pass

        for i in range(self.core_num):
            self.queue.put(None)

        for each in self.processes:
            each.join()

        return results

class AgentType(IntEnum):
    unset = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3
    other = 4

    @staticmethod
    def to_string(a: int):
        return str(AgentType(a)).split('.')[1]
    

args: Args = None
logger = None
eps = 1e-5
origin_point = None
origin_angle = None
index_file = 0
file2pred = {}
files_written = {}
mpl.use('Agg')
visualize_num = 0
traj_last = None
second_span = False
li_vector_num = None
other_errors_dict = defaultdict(list)
_zip = zip
file2targets = {}
i_epoch = None
file2heatmap = {}
ap_list = None
motion_metrics = None
metric_names = None
trajectory_type_2_motion_metrics = {}

def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
time_begin = get_time()

def init(args_: Args, logger_):
    global args, logger
    args = args_
    logger = logger_

    if args.config is not None:
        fs = open(os.path.join('.', 'configs', args.config))
        config = yaml.load(fs, yaml.FullLoader)
        for key in config:
            setattr(args, key, config[key])

    args.output_dir = os.path.join(args.output_dir, time_begin)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.do_eval:
        print(f"args.output_dir: {args.output_dir}")
        assert os.path.exists(args.output_dir)

    if args.log_dir is None:
        args.log_dir = args.output_dir

    # if args.temp_file_dir is None:
    #     args.temp_file_dir = os.path.join(args.output_dir, 'temp_file')
    # else:
    #     args.reuse_temp_file = True
    #     args.temp_file_dir = os.path.join(args.temp_file_dir, 'temp_file')

    dic = {}
    for i, param in enumerate(args.other_params + args.eval_params + args.train_params):
        if '=' in param:
            index = str(param).index('=')
            key = param[:index]
            value = param[index + 1:]
            # key, value = param.split('=')
            dic[key] = value if not str(value).isdigit() else int(value)
        else:
            dic[param] = True
    args.other_params = dic

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.temp_file_dir, exist_ok=True)

    if not args.do_eval and not args.debug:
        src_dir = os.path.join(args.output_dir, 'src')
        if os.path.exists(src_dir):
            subprocess.check_output('rm -r {}'.format(src_dir), shell=True, encoding='utf-8')
        os.makedirs(src_dir, exist_ok=False)
        for each in os.listdir('../src'):
            is_dir = '-r' if os.path.isdir(os.path.join('src', each)) else ''
            # subprocess.check_output(f'cp {is_dir} {os.path.join("src", each)} {src_dir}', shell=True, encoding='utf-8')
        with open(os.path.join(src_dir, 'cmd'), 'w') as file:
            file.write(' '.join(sys.argv))
    args.model_save_dir = os.path.join(args.output_dir, 'model_save')
    os.makedirs(args.model_save_dir, exist_ok=True)

    def init_args_do_eval():
        
        # args.data_dir = 'tf_example/validation/' if not args.do_test else 'tf_example/testing/'
        if 'goals_2D' in args.other_params:
            assert args.nms_threshold is not None or args.method_span[0] > 0 or 'opti' in args.other_params

        # if 'joint_eval' in args.other_params:
        #     args.data_dir = 'tf_example/validation_interactive/'

        if args.model_recover_path is None:
            # args.model_recover_path = os.path.join(args.output_dir, 'model_save', 'model.21.bin')
            args.model_recover_path = os.path.join(args.output_dir, 'model_save', 'model.'+str(args.validation_model)+'.bin')
            
            args_model_recover_path = args.model_path
            
        # elif len(args.model_recover_path) <= 2:
        #     args.model_recover_path = os.path.join(args.output_dir, 'model_save',
        #                                            'model.{}.bin'.format(args.model_recover_path))
        
        
        args.do_train = False
        if len(args.method_span) != 2:
            args.method_span = [args.method_span[0], args.method_span[0] + 1]

        if args.mode_num != 6:
            add_eval_param(f'mode_num={args.mode_num}')

    def init_args_do_train():
        # if 'interactive' in args.other_params:
        #     args.data_dir = 'tf_example/validation_interactive/'
        if args.model_recover_path is None:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save', 'model.21.bin')
        elif len(args.model_recover_path) <= 2:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save',
                                                   'model.{}.bin'.format(args.model_recover_path))
        pass

    if args.do_eval:
        init_args_do_eval()
    else:
        init_args_do_train()

    print(dict(sorted(vars(args_).items())))
    # print(json.dumps(vars(args_), indent=4))
    args_dict = vars(args)
    print()
    logger.info("***** args *****")
    for each in ['output_dir', 'other_params']:
        if each in args_dict:
            temp = args_dict[each]
            if each == 'other_params':
                temp = [param if args.other_params[param] is True else (param, args.other_params[param]) for param in
                        args.other_params]
            print("\033[31m" + each + "\033[0m", temp)
    logging(vars(args_), type='args', is_json=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # os.makedirs(os.path.join(args.temp_file_dir, time_begin), exist_ok=True)

    # if isinstance(args.data_dir, str):
    #     args.data_dir = [args.data_dir]

    assert args.do_train or args.do_eval

    def load_file2targets():
        global file2targets
        if len(file2targets) == 0:
            with open(args.other_params['set_predict_file2targets'], 'rb') as pickle_file:
                file2targets = pickle.load(pickle_file)


def add_eval_param(param):
    if param not in args.eval_params:
        args.eval_params.append(param)


def get_name(name='', append_time=False):
    if name.endswith(time_begin):
        return name
    prefix = 'test.' if args.do_test else 'eval.' if args.do_eval and not args.do_train else ''
    prefix = 'debug.' + prefix if args.debug else prefix
    prefix = args.add_prefix + '.' + prefix if args.add_prefix is not None else prefix
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


def batch_list_to_batch_tensors_old(batch):
    batch_tensors = []
    for x in zip(*batch):
        batch_tensors.append(x)
    return batch_tensors


def round_value(v):
    return round(v / 100)


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def get_dis_point2point(point, point_=(0.0, 0.0)):
    return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))


def get_angle(x, y):
    return math.atan2(y, x)


def get_sub_matrix(traj, object_type, x=0, y=0, angle=None):
    res = []
    for i in range(0, len(traj), 2):
        if i > 0:
            vector = [traj[i - 2] - x, traj[i - 1] - y, traj[i] - x, traj[i + 1] - y]
            if angle is not None:
                vector[0], vector[1] = rotate(vector[0], vector[1], angle)
                vector[2], vector[3] = rotate(vector[2], vector[3], angle)
            res.append(vector)
    return res


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def rotate_(x, y, cos, sin):
    res_x = x * cos - y * sin
    res_y = x * sin + y * cos
    return res_x, res_y


def __iter__(self):  # iterator to load data
    for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
        batch = []
        for __ in range(self.batch_size):
            idx = randint(0, len(self.ex_list) - 1)
            batch.append(self.__getitem__(idx))
        # To Tensor
        yield batch_list_to_batch_tensors(batch)


def logging(*inputs, prob=0.01, type='1', is_json=False, affi=True, sep=' ', to_screen=False, append_time=False, as_pickle=False):
    """
    Print args into log file in a convenient style.
    """
    if to_screen:
        print(*inputs, sep=sep)
    
    if not random.random() <= prob or not hasattr(args, 'log_dir'):
        return

    file = os.path.join(args.log_dir, get_name(type, append_time))
    if as_pickle:
        with open(file, 'wb') as pickle_file:
            assert len(inputs) == 1
            pickle.dump(*inputs, pickle_file)
        return
    if file not in files_written:
        with open(file, "w", encoding='utf-8') as fout:
            files_written[file] = 1
    inputs = list(inputs)
    the_tensor = None
    for i, each in enumerate(inputs):
        if isinstance(each, torch.Tensor):
            # torch.Tensor(a), a must be Float tensor
            if each.is_cuda:
                each = each.cpu()
            inputs[i] = each.data.numpy()
            the_tensor = inputs[i]
    np.set_printoptions(threshold=np.inf)

    with open(file, "a", encoding='utf-8') as fout:
        if is_json:
            for each in inputs:
                print(json.dumps(each, indent=4), file=fout)
        elif affi:
            print(*tuple(inputs), file=fout, sep=sep)
            if the_tensor is not None:
                print(json.dumps(the_tensor.tolist()), file=fout)
            print(file=fout)
        else:
            print(*tuple(inputs), file=fout, sep=sep)
            print(file=fout)


def larger(a, b):
    return a > b + eps


def equal(a, b):
    return True if abs(a - b) < eps else False


def get_valid_lens(matrix: np.ndarray):
    valid_lens = []
    for i in range(matrix.shape[0]):
        ok = False
        for j in range(2, matrix.shape[1], 2):
            if equal(matrix[i][j], 0) and equal(matrix[i][j + 1], 0):
                ok = True
                valid_lens.append(j)
                break

        assert ok
    return valid_lens


def rot(verts, rad):
    rad = -rad
    verts = np.array(verts)
    rotMat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    transVerts = verts.dot(rotMat)
    return transVerts


def visualize_goals_2D(mapping, goals_2D, scores: np.ndarray, future_frame_num, loss=None, labels: np.ndarray = None,
                       labels_is_valid=None, predict: np.ndarray = None, pred_score: np.ndarray = None):
    print('in visualize_goals_2D', mapping['file_name'])
    print('speed', mapping.get('seep', None))
    assert predict is not None
    predict = predict.reshape([6, future_frame_num, 2])
    assert labels.shape == (future_frame_num, 2)

    if 'eval_time' in mapping:
        assert labels.shape[0] == labels_is_valid.shape[0] == future_frame_num
        eval_time = mapping['eval_time']
        labels = labels[:eval_time]
        predict = predict[:, :eval_time, :]
        labels_is_valid = labels_is_valid[:eval_time]
        future_frame_num = eval_time

    if labels_is_valid is not None:
        assert labels.shape[0] == labels_is_valid.shape[0]
        if labels_is_valid.sum()>0:
            labels = [labels[i] for i in range(future_frame_num) if labels_is_valid[i]]
        labels = np.array(labels)

    if 'time_offset' in mapping:
        time_offset = mapping['time_offset']
    else:
        time_offset = None

    assert labels is not None
    labels = labels.reshape([-1])

    fig_scale = 1.0
    marker_size_scale = 4
    # target_agent_color, target_agent_edge_color = '#0d79e7', '#bcd6ed' # blue
    target_agent_color, target_agent_edge_color = '#4bad34', '#c5dfb3'

    def get_scaled_int(a):
        return round(a * fig_scale)

    plt.cla()
    fig = plt.figure(0, figsize=(get_scaled_int(45), get_scaled_int(38)))

    if True:
        plt.xlim(-50, 50)
        plt.ylim(-20, 50)

    # plt.figure(0, dpi=300)
    # cmap = plt.cm.get_cmap('Reds')
    # vmin = np.log(0.0001)
    # vmin = np.log(0.00001)
    # scores = np.clip(scores.copy(), a_min=vmin, a_max=np.inf)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=np.max(scores)))
    # plt.colorbar(sm)

    trajs = mapping['trajs']
    if args.waymo:
        name = mapping['file_name']
    name = name + '.FDE={}'.format(loss)

    add_end = True

    linewidth = 10

    for lane in mapping['vis_lanes']:
        lane = lane[:, :2]
        assert lane.shape == (len(lane), 2), lane.shape
        plt.plot(lane[:, 0], lane[:, 1], linestyle="-", color="black", marker=None,
                 markersize=0,
                 alpha=0.5,
                 linewidth=2,
                 zorder=0)
        # plt.fill(lane[:, 0], lane[:, 1], linestyle="-", color='#a5a5a5',
        #          linewidth=2,
        #          zorder=0)

    yaw_0 = None

    def draw_his_trajs():
        for i, traj in enumerate(trajs):
            assert isinstance(traj, np.ndarray)
            assert traj.ndim == 2 and traj.shape[1] == 2, traj.shape
            if i == 0:
                traj = np.array(traj).reshape([-1])
                t = np.zeros(len(traj) + 2)
                t[:len((traj))] = traj
                t[-2] = labels[0]
                t[-1] = labels[1]

                plt.plot(t[0::2], t[1::2], linestyle="-", color='#df2020', marker=None,
                         alpha=1,
                         linewidth=linewidth,
                         zorder=0)
                # if 'vis_video' in args.other_params:
                # plt.plot(0.0, 0.0, marker=CustomMarker("icon", 0), c=target_agent_color,
                #          markersize=20 * marker_size_scale, markeredgecolor=target_agent_edge_color, markeredgewidth=0.5)
            else:
                if True:
                    pass
                else:
                    if len(traj) >= 2:
                        color = "darkblue"
                        plt.plot(traj[:, 0], traj[:, 1], linestyle="-", color=color, marker=None,
                                 alpha=1,
                                 linewidth=linewidth,
                                 zorder=0)

    draw_his_trajs()

    if True:
        # if goals_2D is not None:
        #     goals_2D = np.array(goals_2D)
        #     marker_size = 70
        #     plt.scatter(goals_2D[:, 0], goals_2D[:, 1], c=scores, cmap=cmap, norm=sm.norm, s=marker_size, alpha=0.5, marker=',')
        # s is size, default 20

        # if False:
        for i in range(len(predict)):
            each = predict[i]
            function2 = plt.plot(each[:, 0], each[:, 1], linestyle="-", color="darkorange", marker=None,
                                 linewidth=linewidth,
                                 zorder=0, label='Predicted trajectory')

            if add_end:
                plt.plot(each[-1, 0], each[-1, 1], markersize=15 * marker_size_scale, color="darkorange", marker="*",
                         markeredgecolor='black')
                # plt.text(each[-1, 0], each[-1, 1]+2, str(i), fontdict={'fontsize': 12})

            # break

        if add_end:
            plt.plot(labels[-2], labels[-1], markersize=15 * marker_size_scale, color=target_agent_color, marker="*",
                     markeredgecolor='black')

        function1 = plt.plot(labels[0::2], labels[1::2], linestyle="-", color=target_agent_color, linewidth=linewidth,
                             zorder=0, label='Ground truth trajectory')

    functions = function1 + function2
    fun_labels = [f.get_label() for f in functions]
    plt.legend(functions, fun_labels, loc=1, fontsize=70)

    # plt.title('FDE={} file_name={}'.format(loss, mapping['file_name']))
    # ax = plt.gca()
    # ax.set_aspect(1)
    # ax.xaxis.set_major_locator(MultipleLocator(4))
    # ax.yaxis.set_major_locator(MultipleLocator(4))
    plt.axis('off')

    os.makedirs(os.path.join(args.log_dir, 'visualize_' + time_begin), exist_ok=True)
    plt.savefig(os.path.join(args.log_dir, 'visualize_' + time_begin,
                             get_name("visualize" + ("" if name == "" else "_" + name) + ".svg")), format='svg', bbox_inches='tight')
    plt.close()
    global visualize_num
    visualize_num += 1
    if visualize_num > 200 and 'vis_video' not in args.other_params and 'vis_all' not in args.other_params:
        print('press any key to continue')
        input()


def load_model(model, state_dict, prefix=''):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix)

    if logger is None:
        return

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, json.dumps(missing_keys, indent=4)))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, json.dumps(unexpected_keys, indent=4)))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def batch_init(mapping):
    global traj_last, origin_point, origin_angle
    batch_size = len(mapping)

    global origin_point, origin_angle
    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']

    def load_file2targets():
        global file2targets
        if len(file2targets) == 0:
            with open(args.other_params['set_predict_file2targets'], 'rb') as pickle_file:
                file2targets = pickle.load(pickle_file)

    def load_file2pred():
        global file2pred
        if len(file2pred) == 0:
            with open(args.other_params['set_predict_file2pred'], 'rb') as pickle_file:
                file2pred = pickle.load(pickle_file)


def merge_tensors(tensors: List[torch.Tensor], device, hidden_size=None) -> Tuple[Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    hidden_size = args.hidden_size if hidden_size is None else hidden_size
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def de_merge_tensors(tensor: Tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]


def merge_tensors_not_add_dim(tensor_list_list, module, sub_batch_size, device):
    # this function is used to save memory in the past at the expense of readability,
    # it will be removed because it is too complicated for understanding
    batch_size = len(tensor_list_list)
    output_tensor_list = []
    for start in range(0, batch_size, sub_batch_size):
        end = min(batch_size, start + sub_batch_size)
        sub_tensor_list_list = tensor_list_list[start:end]
        sub_tensor_list = []
        for each in sub_tensor_list_list:
            sub_tensor_list.extend(each)
        inputs, lengths = merge_tensors(sub_tensor_list, device=device)
        outputs = module(inputs, lengths)
        sub_output_tensor_list = []
        sum = 0
        for each in sub_tensor_list_list:
            sub_output_tensor_list.append(outputs[sum:sum + len(each)])
            sum += len(each)
        output_tensor_list.extend(sub_output_tensor_list)
    return output_tensor_list


def select_goals_by_NMS(mapping: Dict, goals_2D: np.ndarray, scores: np.ndarray, threshold, speed, gt_goal=None, mode_num=6):
    argsort = np.argsort(-scores)
    goals_2D = goals_2D[argsort]
    scores = scores[argsort]

    add_eval_param(f'DY_NMS={threshold}')

    speed_scale_factor = utils_cython.speed_scale_factor(speed)
    threshold = threshold * speed_scale_factor

    pred_goals = []
    pred_probs = []
    pred_indices = []

    def in_predict(pred_goals, point, threshold):
        return np.min(get_dis_point_2_points(point, pred_goals)) < threshold

    for i in range(len(goals_2D)):
        if len(pred_goals) > 0 and in_predict(np.array(pred_goals), goals_2D[i], threshold):
            continue
        else:
            pred_goals.append(goals_2D[i])
            pred_probs.append(scores[i])
            pred_indices.append(i)
            if len(pred_goals) == mode_num:
                break

    while len(pred_goals) < mode_num:
        i = np.random.randint(0, len(goals_2D))
        pred_goals.append(goals_2D[i])
        pred_probs.append(scores[i])
        pred_indices.append(i)

    pred_goals = np.array(pred_goals)
    pred_probs = np.array(pred_probs)

    FDE = np.inf
    if gt_goal is not None:
        for each in pred_goals:
            FDE = min(FDE, get_dis_point2point(each, gt_goal))

    mapping['pred_goals'] = pred_goals
    mapping['pred_probs'] = pred_probs
    mapping['pred_indices'] = pred_indices


def select_goal_pairs_by_NMS(mapping: Dict, mapping_oppo: Dict, goals_4D: np.ndarray, scores_4D: np.ndarray, threshold, speed, speed_oppo, args,
                             mode_num=6):
    argsort = np.argsort(-scores_4D)

    goals_4D = goals_4D[argsort]
    scores_4D = scores_4D[argsort]

    def in_predict(pred_goal_pairs, goal_pair, thresholds, args):
        if args.joint_nms_type == "or":
            valid_goal = np.min(get_dis_point_2_points(goal_pair[0], pred_goal_pairs[:, 0, :])) < thresholds[0] \
                         or np.min(get_dis_point_2_points(goal_pair[1], pred_goal_pairs[:, 1, :])) < thresholds[1]
        else:
            valid_goal = np.min(get_dis_point_2_points(goal_pair[0], pred_goal_pairs[:, 0, :])) < thresholds[0] \
                         and np.min(get_dis_point_2_points(goal_pair[1], pred_goal_pairs[:, 1, :])) < thresholds[1]
        return valid_goal

    add_eval_param(f'DY_NMS={threshold}')

    thresholds = (threshold * utils_cython.speed_scale_factor(speed), threshold * utils_cython.speed_scale_factor(speed_oppo))

    pred_goal_pairs = []
    pred_probs = []

    for i in range(len(goals_4D)):
        if len(pred_goal_pairs) > 0 and in_predict(np.array(pred_goal_pairs), goals_4D[i].reshape((2, 2)), thresholds, args):
            continue
        else:
            pred_goal_pairs.append(goals_4D[i].reshape((2, 2)))
            pred_probs.append(scores_4D[i])
            if len(pred_goal_pairs) == mode_num:
                break

    while len(pred_goal_pairs) < mode_num:
        i = np.random.randint(0, len(goals_4D))
        pred_goal_pairs.append(goals_4D[i].reshape((2, 2)))
        pred_probs.append(scores_4D[i])

    pred_goal_pairs = np.array(pred_goal_pairs)
    pred_probs = np.array(pred_probs)

    mapping['pred_goals'] = pred_goal_pairs[:, 0, :]
    mapping['pred_probs'] = pred_probs
    mapping_oppo['pred_goals'] = pred_goal_pairs[:, 1, :]
    mapping_oppo['pred_probs'] = pred_probs


def assert_(satisfied, info=None):
    if not satisfied:
        if info is not None:
            print(info)
        print(sys._getframe().f_code.co_filename, sys._getframe().f_back.f_lineno)
    assert satisfied


def get_miss_rate(li_FDE, dis=2.0):
    return np.sum(np.array(li_FDE) > dis) / len(li_FDE) if len(li_FDE) > 0 else None



def other_errors_put(error_type, error):
    other_errors_dict[error_type].append(error)


def other_errors_to_string():
    res = {}
    for each, value in other_errors_dict.items():
        res[each] = np.mean(value)
    return str(res)


def get_dis_point_2_points(point, points):
    assert points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))



def zip(*inputs):
    for each in inputs:
        assert len(each) == len(inputs[0])
    return _zip(*inputs)


def zip_enum(*inputs):
    for each in inputs:
        assert len(each) == len(inputs[0])
    return zip(range(len(inputs[0])), *inputs)


def get_from_mapping(mapping: List[Dict], key=None):
    if key is None:
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]



def metric_values_to_string(metric_values, metric_names, metric=None, index=None, append=False):
    if metric_values == None:
        print('metric_values is None')
        return
    lines = []
    for i, m in enumerate(
            ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
        if metric is None or metric == m:
            for j, n in enumerate(metric_names):
                if index is None or index == j:
                    if append and metric_values[i][j] > 0.0:
                        ap_list.append(float(metric_values[i][j]))
                    lines.append('{}/{}: {}'.format(m, n, metric_values[i][j]))
    return '\n'.join(lines)


def pool_forward(rank, queue, result_queue, run):
    while True:
        file = queue.get()
        if file is None:
            break
        result = run(*file)
        result_queue.put(result)


def get_eval_identifier():
    eval_identifier = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15 and '=' in each:
            each = each.split('=')[0]
        if len(each) > 15:
            each = 'long'
        eval_identifier += '.' + str(each)
    eval_identifier = get_name(eval_identifier, append_time=True)
    return eval_identifier


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def type_is_ok(type, args):
    return args.agent_type is None or type == AgentType[args.agent_type]


def types_are_ok(types, args):
    return args.inter_agent_types is None or \
           types[0] == AgentType[args.inter_agent_types[0]] and types[1] == AgentType[args.inter_agent_types[1]]
           
           
def load_scenario_from_dictionary(dictionary_to_load, scenario_id):
    if scenario_id in dictionary_to_load.keys():
        return dictionary_to_load[scenario_id]
    elif bytes.decode(scenario_id) in dictionary_to_load.keys():
        return dictionary_to_load[bytes.decode(scenario_id)]
    else:
        return None
    
    


def get_influencer_idx(objects_id, scenario_id):
    """
    check if current target agent in directR files, if no, return None to skip
    if yes, return its influencer index (return 0 if agent has no influcner)
    """

    selected_agent_id = int(objects_id[0])
    # get current_agent's influencer, if there is any
    direct_relation_dic = globals.direct_relation
    assert direct_relation_dic is not None
    direct_relation = load_scenario_from_dictionary(direct_relation_dic, scenario_id)
    if direct_relation is None:
        # skip not found scenarios
        return None
    direct_relation = np.array(direct_relation)
    if len(direct_relation.shape) == 1:
        direct_relation = direct_relation[np.newaxis, :]


    # return influencers_indices
    influencer_indices = []
    for influencer_id, reactor_id, relation_label in direct_relation:
        if reactor_id == selected_agent_id:
            if relation_label != 2:
                for idx, agent_id in enumerate(objects_id):
                    if int(agent_id) == influencer_id:
                        influencer_indices.append(idx)
    return influencer_indices
