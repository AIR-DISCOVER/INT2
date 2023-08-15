import math
import os
import pickle
import random
from functools import partial
import copy
import numpy as np
import torch
from tqdm import tqdm
from . import globals, structs, utils, utils_cython
from .waymo_tutorial import _parse
from waymo_open_dataset.protos import motion_submission_pb2
from collections import defaultdict
import tensorflow as tf
Normalizer = utils.Normalizer
from enum import IntEnum
from IPython import embed

tqdm = partial(tqdm, dynamic_ncols=True)


loading_summary = {
    'all_scenarios': 0,
    'scenarios_in_traj_pred': 0,
    'two_agents_found_in_traj_pred': 0,
    'scenarios_in_relation_gt': 0,
    'scenarios_in_relation_pred': 0
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, rank=0, to_screen=True):
        # self.loader = WaymoDL(args.data_dir[0])
        self.args = args
        self.rank = rank
        self.world_size = args.distributed_training if args.distributed_training else 1
        self.batch_size = batch_size

        self.file_names = [i[:-1] for i in open(args.data_txt)]
        self.file_names = sorted(self.file_names)

        if to_screen:
            print("valid file_names is", len(self.file_names))

        if args.do_eval:
            self.load_queue = np.arange(len(self.file_names))
            self.load_queue = self.load_queue[self.rank::self.world_size]
            self.load_queue = iter(self.load_queue)
            self.waymo_generate(expected_len=400)
        else:
            self.set_epoch(0)
            
        self.batch_size = batch_size

        self.loading_summary = loading_summary                                  

    def __len__(self):
        args = self.args
        if self.args.do_eval:
            # return int(500_000 * 0.15 * 2 / self.world_size)
            return int(10000 / self.world_size)
        else:
            # 除以长度之后就是在除以batch size之后的数据长度，每个设备中这个长度都是一样的 10937
            # 左边数据的长度就是多张卡总共要跑完的数据量
            
            # 500_000 for vv
            # 300_000 for vc
            # 200_000 for vp
            return int(20000 * 0.7 / self.batch_size)
            # return int(500_000 * 0.7 / self.batch_size) 

    def __getitem__(self, idx):
        # print('__getitem__', idx)
        return self.__next__()

    def __next__(self):
        if isinstance(self.ex_list, list):
            self.ex_list = iter(self.ex_list)

        if self.args.do_eval:
            try:
                return next(self.ex_list)
            except StopIteration:
                return None
        else:
            mapping = []
            # batch size 表示全局的batch size
            for i in range(self.batch_size // self.world_size):
                try:
                    mapping.append(next(self.ex_list))
                except StopIteration:
                    return None

            return mapping

    def set_epoch(self, i_epoch):
        if i_epoch == 0:
            if hasattr(self, 'load_queue'):
                return
        self.load_queue = np.arange(len(self.file_names))
        np.random.seed(i_epoch)
        np.random.shuffle(self.load_queue)
        utils.logging('set_train_epoch', self.load_queue[:20])
        # 从总数据中根据设备的id获取对应的独一无二的数据
        # 这样看起来每个设备上面的数据就都是一样的了
        self.load_queue = self.load_queue[self.rank::self.world_size]
        self.load_queue = iter(self.load_queue)
        self.waymo_generate(expected_len=400)
        self.set_ex_list_length(400)
        # self.waymo_generate(expected_len=200)
        # self.set_ex_list_length(200)

    def set_ex_list_length(self, length):
        if self.args.do_train:
            random.shuffle(self.ex_list)
            if not self.args.debug_mode:
                assert len(self.ex_list) >= length, str(len(self.ex_list)) + '/' + str(length)
            self.ex_list = self.ex_list[:length]

    def waymo_generate(self, expected_len=200 * 50):
        self.ex_list = []

        args = self.args

        if args.do_eval:
            expected_len = 300

        if 'raster' in args.other_params and args.do_train:
            if expected_len > 500 * 10:
                expected_len = expected_len // 2

        assert expected_len >= self.batch_size

        while len(self.ex_list) < expected_len:
            try:
                file_name = self.file_names[next(self.load_queue)]
            except StopIteration:
                return False, len(self.ex_list)

            self.ex_list.extend(get_ex_list_from_file(file_name, args))

        random.shuffle(self.ex_list)

        return True, len(self.ex_list)


def get_ex_list_from_file(file_name, args: utils.Args, trajectory_type_2_ex_list=None, balance_queue=None):
    ex_list = []
    dataset = np.load(file_name, allow_pickle=True)

    # dataset = np.load(file_name, allow_pickle=True)
    # for step, data in enumerate(dataset):
    for step in range(len(dataset)):
        # inputs, decoded_example = _parse(data)
        inputs, decoded_example = _parse(dataset[step])
        
        # get predict_agent_num
        sample_is_valid = inputs['sample_is_valid']
        if 'pred_all_agents' in args.other_params:
            tracks_to_predict = sample_is_valid
        else:
            tracks_to_predict = inputs['tracks_to_predict'][sample_is_valid]
        # Set predicting tracks to interactive tracks if train_pair_interest flag is specified.
        if 'train_pair_interest' in args.other_params:
            interactive_tracks_to_predict = inputs['interactive_tracks_to_predict'][sample_is_valid]
            tracks_to_predict = tracks_to_predict & interactive_tracks_to_predict

        predict_agent_num = tracks_to_predict.sum()
        tracks_type = decoded_example['state/type'][sample_is_valid]
        tracks_type = tracks_type.copy().reshape(-1)

        if args.do_eval:
            instance = []
            if 'joint_eval' in args.other_params:
                assert predict_agent_num == 2
                if utils.types_are_ok((tracks_type[0], tracks_type[1]), args) or utils.types_are_ok((tracks_type[1], tracks_type[0]), args):
                    for select in range(predict_agent_num):
                        t = get_instance(args, inputs, decoded_example, f'{os.path.split(file_name)[1]}.{str(step)}',
                                         select=select)
                        assert t is not None
                        instance.append(t)
            else:
                if 'train_pair_interest' in args.other_params:
                    assert predict_agent_num == 2, predict_agent_num
                for select in range(predict_agent_num):
                    # Make sure both agents are of the same type if it is specified.
                    # if type_is_ok(tracks_type[select], args) and type_is_ok(tracks_type[1 - select], args):
                    if True:
                        t = get_instance(args, inputs, decoded_example, f'{os.path.split(file_name)[1]}.{str(step)}',
                                         select=select)
                                         
                        # change to a soft check when eval
                        if t is not None:
                            instance.append(t)
                        # assert t is not None
                        # instance.append(t)
                if 'train_pair_interest' in args.other_params:
                    assert len(instance) in [0, 1, 2], len(instance)

            if len(instance) > 0:
                ex_list.append(instance)
        else:
            if 'train_pair_interest' in args.other_params:
                instance = [get_instance(args, inputs, decoded_example, f'{os.path.split(file_name)[1]}.{str(step)}',
                                         select=select)
                            for select in range(2)]
                if None not in instance:
                    ex_list.append(instance)
            else:
                instance = get_instance(args, inputs, decoded_example, f'{os.path.split(file_name)[1]}.{str(step)}')

                if instance is not None:
                    ex_list.append(instance)

        # if step > 200:
        #     break

    return ex_list


def extract_from_inputs(inputs, decoded_example, args, select, idx_in_K):
    sample_is_valid = inputs['sample_is_valid']
    gt_trajectory = inputs['gt_future_states'][sample_is_valid]
    gt_future_is_valid = inputs['gt_future_is_valid'][sample_is_valid]
    tracks_to_predict = inputs['tracks_to_predict'][sample_is_valid]
    tracks_type = decoded_example['state/type'][sample_is_valid]
    objects_id = decoded_example['state/id'][sample_is_valid]


    scenario_id = decoded_example['scenario/id'] + '_' + str(int(decoded_example['state/id'][0])) + '_' + str(int(decoded_example['state/id'][1])) + '_' + str(np.array(decoded_example['state/current/timestamp_micros'][0])[0])

    # For interactive dataset, map select from indices to [0, 1], since interactive agents are not always the first 2 agents.
    if 'train_pair_interest' in args.other_params:
        objects_of_interest = decoded_example['state/objects_of_interest'][sample_is_valid]
        indices = np.nonzero(objects_of_interest)[0]
        if select == 0:
            select = indices[0]
        elif select == 1:
            select = indices[1]
        else:
            raise NotImplementedError

    mapping_eval = {
        'gt_trajectory': gt_trajectory[select],
        'gt_is_valid': gt_future_is_valid[select],
        'object_type': tracks_type[select],
        'object_id': objects_id[select],
        'scenario_id': scenario_id,

        'idx_in_predict_num': select,
        'idx_in_K': idx_in_K,
    } if args.do_eval else None

    gt_trajectory = gt_trajectory.copy()
    gt_future_is_valid = gt_future_is_valid.copy()
    tracks_to_predict = tracks_to_predict.copy()
    tracks_type = tracks_type.copy().reshape(-1)
    
    # 将influencer和reactor的type都设置为1
    tracks_type[0] = 1
    tracks_type[1] = 1
    objects_id = objects_id.copy()

    predict_agent_num = tracks_to_predict.sum()

    '''
    when prediction relation, the number of predict agent num is 2.
    when prediction marginal, the number of predict agent num is 2.
    when prediction conditional, the number of predict agent num always is 2.
    
    '''
    for i in range(predict_agent_num):
        assert tracks_to_predict[i]
    assert len(gt_trajectory) == len(gt_future_is_valid) == len(tracks_type)

    return sample_is_valid, gt_trajectory, gt_future_is_valid, tracks_to_predict, tracks_type, objects_id, scenario_id, mapping_eval


def get_instance(args: utils.Args, inputs, decoded_example, file_name,
                 select=None, time_offset=None, idx_in_K=None):
    sample_is_valid, gt_trajectory, gt_future_is_valid, tracks_to_predict, tracks_type, objects_id, scenario_id, mapping_eval = \
        extract_from_inputs(inputs, decoded_example, args, select, idx_in_K)

    predict_agent_num = tracks_to_predict.sum()
    mapping_before = {}
    # use future_frame_num as default value for eval_time
    eval_time = args.other_params.get('eval_time', args.future_frame_num)
    history_frame_num = 11

    if time_offset is not None:
        if time_offset > 0:
            gt_trajectory = np.concatenate([gt_trajectory[:, time_offset:, :], gt_trajectory[:, :time_offset, :]], axis=1)
            gt_future_is_valid = np.concatenate([gt_future_is_valid[:, time_offset:], gt_future_is_valid[:, :time_offset]], axis=1)
        file_name = f'{file_name}.time_offset={time_offset:02d}'
        mapping_eval['time_offset'] = time_offset

    if True:
        whole_final_idx_eval = -1 if eval_time == 80 else history_frame_num + eval_time - 1
        whole_final_idx_training = -1 if args.future_frame_num == 80 else history_frame_num + args.future_frame_num - 1
        # the files loading process: T_gt(outside of this function) -> T_p -> R_gt -> R_p
        # load marginal prediction result
        loading_summary['all_scenarios'] += 1

        if args.influencer_pred_file_path is not None:
            if globals.influencer_pred is None:
                print("loading trajectory prediction from: ", args.influencer_pred_file_path)
                globals.influencer_pred = structs.load(args.influencer_pred_file_path)
                print("pd trajectory loaded")
            # replace gt with marginal predictions
            influencer_pred = globals.influencer_pred
            loaded_inf = utils.load_scenario_from_dictionary(influencer_pred, scenario_id)
            if loaded_inf is not None:
                prediction_result = loaded_inf['rst']
                agents_ids_in_prediction = loaded_inf['ids']
                prediction_scores = loaded_inf['score']
            else:
                return None
            loading_summary['scenarios_in_traj_pred'] += 1
        # load relation
        if args.relation_file_path is not None:
            if args.do_test:
                # when predicting against test dataset, dummify the interaction_label for computing relation error
                interaction_label = 0
            else:
                # if globals.interactive_relations is None:
                #     print("loading relation gt from: ", args.relation_file_path)
                #     # globals.interactive_relations = structs.load(args.relation_file_path)
                #     globals.interactive_relations = {'yizhuang#1/1650251580.01-1650251760.00': np.array([1535178, 1535205, 0, 1])}
                #     print("loading finished")
                # interactive_relations = globals.interactive_relations
                # relation = utils.load_scenario_from_dictionary(interactive_relations, scenario_id)
                relation = decoded_example['relation']

                if relation is None:
                    return None
                # assert len(relation) == 4, "Relation data should include 4 elements."
                # [id1, id2, label, relation_type].
                # id1 always influencer, id2 always reactor
                # interaction_label: 1 - bigger id agent dominant, 0 - smaller id agent dominant, 2 - no relation
                # agent_pair_label: 1 - v2v, 2 - v2p, 3 - v2c, 4 - others
                id1, id2, interaction_label, agent_pair_label = relation[:4]
                
                # 无论训练vv vc 还是 vp，都以vv的形式来训练
                agent_pair_label = 1
                
                # You can only train one type or all types
                if 'pair_vv' in args.other_params:
                    # Train with v2v data
                    if agent_pair_label != 1:
                        return None
                elif 'pair_vp' in args.other_params:
                    # Train with v2p data
                    if agent_pair_label != 2:
                        return None
                elif 'pair_vc' in args.other_params:
                    # Train with v2c data
                    if agent_pair_label != 3:
                        # return None
                        pass
                elif 'pair_others' in args.other_params:
                    # Train with other type data
                    if agent_pair_label != 4:
                        return None
                # You can train with two additional modes
                if 'binary_is_two' in args.other_params:
                    interaction_label = 0 if interaction_label < 2 else 1
                if '0and1' in args.other_params and interaction_label == 2:
                    return None
                if interaction_label == 2:
                    # in label 2, the ids are from previous scenario
                    id1 = id2 = None
                else:
                    id1 = int(id1)
                    id2 = int(id2)

                loading_summary['scenarios_in_relation_gt'] += 1
        
        if 'direct_relation_label' in args.other_params:
            assert args.direct_relation_path is not None, f'pass direct relation file path to use'
            # this is the new direct type of relation, a list of [influencer_id, reactor_id]
            # sample:  [[25.0, 1.0], [1.0, 2.0], [1.0, 4.0], [2.0, 8.0], [1.0, 9.0]]
            # new sample:  [[25.0, 1.0, 1], [1.0, 2.0, 2], [1.0, 4.0, 0], [2.0, 8.0, 2], [1.0, 9.0, 1]]
            if globals.direct_relation is None:
                print("loading direct relation from: ", args.direct_relation_path)
                globals.direct_relation = structs.load(args.direct_relation_path)
                print("loading direct relation finished")
    
        # load relationship prediction result and filter
        if 'gt_relation_label' in args.other_params:
            relation_label_pred = interaction_label
        elif args.relation_pred_file_path is not None:
            if globals.relation_pred is None:
                print("loading relation prediction from: ", args.relation_pred_file_path)
                globals.relation_pred = structs.load(args.relation_pred_file_path)
                print("loading pd relation finished")
            relation_pred = globals.relation_pred
            relation_pred_rst = utils.load_scenario_from_dictionary(relation_pred, scenario_id)
            if relation_pred_rst is None:
                return None
            if isinstance(relation_pred_rst, int):
                relation_label_pred = relation_pred_rst
            elif isinstance(relation_pred_rst, type([])):
                # for [result, score]
                relation_label_pred, score = relation_pred_rst
            else:
                print("unrecognized relation loaded: ", relation_pred_rst)
                return None
        else:
            relation_label_pred = None

            loading_summary['scenarios_in_traj_pred'] += 1

        ##################  End of files loading  ##################

        if 'train_interest' in args.other_params and args.do_train:
            assert select is None
            if not move_interest_objects_forward(decoded_example, sample_is_valid, gt_trajectory,
                                                 gt_future_is_valid, tracks_type, objects_id, args):
                return
        elif 'train_pair_interest' in args.other_params:
            assert select in [0, 1]
            if not move_interest_objects_forward(decoded_example, sample_is_valid, gt_trajectory,
                                                 gt_future_is_valid, tracks_type, objects_id, args, keep_order=True):
                return
            if select == 1:
                # warning: swap all
                def swap(tensor):
                    tensor[0], tensor[select] = tensor[select].copy(), tensor[0].copy()

                for each in [gt_trajectory, gt_future_is_valid, tracks_type, objects_id]:
                    swap(each)
        elif 'direct_relation_label' in args.other_params:
            influencer_indices = utils.get_influencer_idx(objects_id, scenario_id)
            if influencer_indices is None:
                return None
            else:
                if len(influencer_indices) == 0:
                    influencer_labels = [0]
                else:
                    influencer_labels = influencer_indices
                # encoding indices with binary labels
                inf_labels = np.zeros((128))
                for idx in influencer_labels:
                    inf_labels[idx] = 1
                assert np.sum(inf_labels) > 0, inf_labels
                mapping_before['influencer_idx'] = inf_labels
        elif 'train_relation' in args.other_params:
            if 'save_rst' in args.other_params:
                # override the save_rst, when eval utils will automatically save the result
                args.other_params.remove('save_rst')
            else:
                if not move_interest_objects_forward(decoded_example, sample_is_valid, gt_trajectory,
                                                     gt_future_is_valid, tracks_type, objects_id, args, order_by_id=1):
                    return

                # store gt relation for loss computation
                mapping_before['interaction_label'] = interaction_label

        # training reactor following Influencer->Reactor
        elif 'train_reactor' in args.other_params:
            # new code for reactor for filtering invalid samples
            if select is None:
                
                # warning: gt_future_is_valid[i, -1]
                final_valid_ids = [i for i in range(predict_agent_num)
                                   if (gt_future_is_valid[i, whole_final_idx_training] or 'allow_2' in args.other_params)
                                   and utils.type_is_ok(tracks_type[i], args)]
                
                if len(final_valid_ids) == 0:
                    return None
                select = random.choice(final_valid_ids) if len(final_valid_ids) > 0 else 0
                
            else:
                if not gt_future_is_valid[select, whole_final_idx_training] and args.do_train:
                    return None



            if 'train_from_large' in args.other_params:
                assert 'gt_influencer_traj' in args.other_params, 'train from large must use gt-traj mode'
                gt_influencer = np.zeros(gt_trajectory[1].shape, dtype=np.float32)
                # warning: swap all
                def swap(tensor):
                    tensor[0], tensor[select] = tensor[select].copy(), tensor[0].copy()

                for each in [gt_trajectory, gt_future_is_valid, tracks_type, objects_id]:
                    swap(each)
            else:
                
                if relation_label_pred < 2:  # relation_label_red is vv
                    if args.reverse_pred_relation:
                        relation_label_pred = 1 - relation_label_pred
                    # move predicted reactor as the first one to predict
                    
                    if not move_interest_objects_forward(decoded_example, sample_is_valid, gt_trajectory,
                                                         gt_future_is_valid, tracks_type, objects_id,
                                                         args, order_by_id=1-relation_label_pred+1):
                        return
                    
                    influencer_agent_id = int(objects_id[1])
                    reactor_agent_id = int(objects_id[0])
                    if relation_label_pred == 0:
                        assert influencer_agent_id < reactor_agent_id
                    elif relation_label_pred == 1:
                        assert influencer_agent_id > reactor_agent_id
                    else:
                        assert False, f'loaded relation not recognized {relation_label_pred}'
                    if args.do_eval:
                        # skipping evaluation for influencer
                        if int(mapping_eval['object_id']) != int(reactor_agent_id):
                            return None
                else:
                    # skip prediction for label 2 (use marginal prediction result instead)
                    return None
                
                
                if args.influencer_pred_file_path is not None:
                    # clean prediction result for influencer only
                    assert len(prediction_result.shape) in [3, 4], prediction_result.shape
                    if len(prediction_result.shape) == 3:
                        assert prediction_result.shape == (6, 80, 2), prediction_result.shape
                        print("Your influencer prediction file has only one agent in it, shape: (6, 80, 2)")
                    num_of_agents_in_prediction, _, _, _ = prediction_result.shape
                    prediction_result_inf = None
                    for i in range(num_of_agents_in_prediction):
                        if agents_ids_in_prediction[i] == influencer_agent_id:
                            prediction_result_inf = prediction_result[i]
                            prediction_scores_inf = prediction_scores[i]
                    assert prediction_result_inf is not None, f'{influencer_agent_id} not found in {agents_ids_in_prediction} at {scenario_id} with {id1},{id2}, {objects_id[:2]}'
                    assert prediction_result_inf.shape == (6, 80, 2), prediction_result_inf.shape
                    assert prediction_scores_inf.shape == (6,), prediction_scores.shape


                # smuggle gt trajectory of the influencer into vector
                influencer_pred_score = None
                if 'gt_influencer_traj' not in args.other_params:
                    # use predictions for training and evaluation
                    _, t, xyect = gt_trajectory.shape
                    # smuggle predicted trajectory of the influencer into vector
                    if args.eval_rst_saving_number is not None:
                        influencer_pred_rst = np.zeros((1, t, xyect), dtype=np.float32)
                        # for models trained with Trajectory_gt, load one prediction per time when doing inference
                        # change target_idx to load different influencer prediction result from 0 to 5
                        target_idx = int(args.eval_rst_saving_number)
                        influencer_pred_rst[0, :, :2] = np.concatenate(
                            [gt_trajectory[1, :11, :2], prediction_result_inf[target_idx]])
                        # include all infos in the past
                        influencer_pred_rst[0, :11, :] = gt_trajectory[1, :11, :].copy()
                        influencer_pred_score = np.array(prediction_scores_inf[target_idx], dtype=np.float32)
                    else:
                        influencer_pred_rst = np.zeros((6, t, xyect), dtype=np.float32)
                        # gives 6 gt_traj as prediction
                        k_gt_trajectory = np.array(
                            [gt_trajectory[1], gt_trajectory[1], gt_trajectory[1], gt_trajectory[1], gt_trajectory[1],
                             gt_trajectory[1]], dtype=np.float32)
                        assert len(k_gt_trajectory.shape) == 3, k_gt_trajectory.shape
                        influencer_pred_rst[:, :, :2] = np.concatenate([k_gt_trajectory[:, :11, :2], prediction_result_inf],
                                                                       axis=1).copy()
                        # include all infos in the past
                        influencer_pred_rst[:, :11, :] = k_gt_trajectory[:, :11, :].copy()
                        influencer_pred_score = np.array(prediction_scores_inf, dtype=np.float32)
                    # rescale to sum as 1
                    influencer_pred_score = np.exp(influencer_pred_score)
                    influencer_pred_score = influencer_pred_score / np.sum(influencer_pred_score)
                else:
                    gt_influencer = gt_trajectory[1].copy()
                    # to keep input the same, only include yaw infos before frame 11
                    gt_influencer[11:, 2:] = 0
                    if args.eval_rst_saving_number is None:
                        # use gt_traj for training: when training, do not give eval_rst_saving_number, leave gt_traj as it is
                        pass
                    else:
                        # use one of the prediction for evaluation, send eval_rst_saving_number
                        target_idx = int(args.eval_rst_saving_number)
                        gt_influencer[:, :2] = np.concatenate(
                            [gt_trajectory[1, :11, :2], prediction_result_inf[target_idx]])
                        gt_influencer[:11, :] = gt_trajectory[1, :11, :].copy()
       
       
        else:
            if select is None:
                # warning: gt_future_is_valid[i, -1]
                final_valid_ids = [i for i in range(predict_agent_num)
                                   if (gt_future_is_valid[i, whole_final_idx_training] or 'allow_2' in args.other_params)
                                   and utils.type_is_ok(tracks_type[i], args)]
                if len(final_valid_ids) == 0:
                    return None
                select = random.choice(final_valid_ids) if len(final_valid_ids) > 0 else 0
            else:
                if not gt_future_is_valid[select, whole_final_idx_eval] and args.do_train:
                    return None

            # warning: swap all
            def swap(tensor):
                tensor[0], tensor[select] = tensor[select].copy(), tensor[0].copy()

            # Swap selected agent to index 0.
            # 调换位置
            for each in [gt_trajectory, gt_future_is_valid, tracks_type, objects_id]:
                swap(each)

            # check relevant agents for marginal predictions
            if 'pred_all_agents' in args.other_params:
                assert args.all_agent_ids_path is not None
                if globals.all_relevant_agent_ids is None:
                    print('loading relevant agent ids')
                    globals.all_relevant_agent_ids = structs.load(args.all_agent_ids_path)
                agent_ids_relevant = utils.load_scenario_from_dictionary(globals.all_relevant_agent_ids, scenario_id)
                if agent_ids_relevant is None:
                    # skip not included scenarios
                    return None
                if int(objects_id[0]) not in agent_ids_relevant:
                    # skip not relevant agents
                    return None

        select = None

        if not gt_future_is_valid[0, history_frame_num - 1]:
            return None

        last_valid_index = history_frame_num - 1

        speed = utils.get_dis_point2point((gt_trajectory[0, history_frame_num - 1, 5], gt_trajectory[0, history_frame_num - 1, 6]))
        waymo_yaw = gt_trajectory[0, last_valid_index, 4]
        track_type_int = tracks_type[0]
        trajectory_type = utils_cython.classify_track(gt_future_is_valid[0], gt_trajectory[0])
        headings = gt_trajectory[0, history_frame_num:, 4].copy()

        angle = -waymo_yaw + math.radians(90)

        normalizer = utils.Normalizer(gt_trajectory[0, last_valid_index, 0], gt_trajectory[0, last_valid_index, 1], angle)

        # _gt_trajectory = copy.deepcopy(gt_trajectory)
        gt_trajectory[:, :, :2] = utils_cython.get_normalized(gt_trajectory[:, :, :2], normalizer)
        if 'train_reactor' in args.other_params:
            if 'gt_influencer_traj' in args.other_params:
                gt_influencer[:, :2] = utils_cython.get_normalized(gt_influencer[:, :2][np.newaxis, :], normalizer)[0]
            else:
                influencer_pred_rst[:, :, :2] = utils_cython.get_normalized(influencer_pred_rst[:, :, :2], normalizer)
        if 'relation_wpred' in args.other_params:
            gt_trajectory_pair[0, :, :, :2] = utils_cython.get_normalized(gt_trajectory_pair[0, :, :, :2], normalizer)
            gt_trajectory_pair[1, :, :, :2] = utils_cython.get_normalized(gt_trajectory_pair[1, :, :, :2], normalizer)

        labels = gt_trajectory[0, history_frame_num:history_frame_num + args.future_frame_num, :2].copy() * \
                 gt_future_is_valid[0, history_frame_num:history_frame_num + args.future_frame_num, np.newaxis]

        yaw_labels = gt_trajectory[0, history_frame_num:history_frame_num + args.future_frame_num, 2].copy() * \
                     gt_future_is_valid[0, history_frame_num:history_frame_num + args.future_frame_num]

        labels_is_valid = gt_future_is_valid[0, history_frame_num:history_frame_num + args.future_frame_num].copy()

    if 'raster' in args.other_params:
        if 'train_reactor' in args.other_params and 'raster_inf' in args.other_params:
            image = np.zeros([224, 224, 60 + 90], dtype=np.int8)
        else:
            image = np.zeros([224, 224, 60], dtype=np.int8)
        args.image = image

    # check target agent type
    if not utils.type_is_ok(tracks_type[0], args):
        return None

    if 'train_reactor' in args.other_params and 'raster_inf' in args.other_params:
        if 'gt_influencer_traj' in args.other_params:
            vectors, polyline_spans, trajs = utils_cython.get_agents(gt_trajectory, gt_future_is_valid, tracks_type,
                                                                     args.visualize, args,
                                                                     gt_influencer)
        else:
            vectors, polyline_spans, trajs = utils_cython.get_agents(gt_trajectory, gt_future_is_valid, tracks_type,
                                                                     args.visualize, args,
                                                                     influencer_pred_rst, influencer_pred_score)
    else:
        vectors, polyline_spans, trajs = utils_cython.get_agents(gt_trajectory, gt_future_is_valid, tracks_type, args.visualize, args)

    map_start_polyline_idx = len(polyline_spans)

    vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(decoded_example, normalizer, args)
    
    polyline_spans_ = polyline_spans_ + len(vectors)
    vectors = np.concatenate([vectors, vectors_])
    polyline_spans = np.concatenate([polyline_spans, polyline_spans_])

    gt_influencer_traj_idx = len(polyline_spans)
    if args.infMLP > 0:
        # time_steps = gt_trajectory.shape[0]
        time_steps = 91
        if 'gt_influencer_traj' in args.other_params:
            if 'transpose_inf' in args.other_params:
                # gt_influencer_vector: [2, 128]
                gt_influencer_vector = np.zeros((2, vectors.shape[1]))
                gt_influencer_vector[:, :91] = gt_influencer[:, :2].transpose()
                yaw_vector = np.zeros((1, vectors.shape[1]))
                size_vector = np.zeros((2, vectors.shape[1]))
                vectors = np.concatenate([vectors, gt_influencer_vector])
                # update span
                for _ in range(2):
                    last_span_idx = polyline_spans[-1, 1]
                    new_span = np.array([last_span_idx, last_span_idx + 1]).reshape(1, 2)
                    polyline_spans = np.concatenate([polyline_spans, new_span])
            else:
                # stack gt_influencer prediction/gt result into vectors
                # gt_trajectory has 91 length, vector has 128 shape on 1
                gt_influencer_vector = np.zeros((time_steps, vectors.shape[1]))
                if 'train_reactor' in args.other_params:
                    # assert time_steps == gt_influencer.shape[0], f'{time_steps} / {gt_influencer.shape} / {gt_trajectory.shape}'
                    gt_influencer_vector[:, :2] = gt_influencer[:, :2]
                vectors = np.concatenate([vectors, gt_influencer_vector])
                # update span
                last_span_idx = polyline_spans[-1, 1]
                new_span = np.array([last_span_idx, last_span_idx + time_steps]).reshape(1, 2)
                polyline_spans = np.concatenate([polyline_spans, new_span])
        else:
            gt_trajectory_pair = np.array([[gt_trajectory[0].copy() for _ in range(6)],
                                           [gt_trajectory[1].copy() for _ in range(6)]],
                                          dtype=np.float32)  # [2, 6, 91, 2]
            if 'relation_wpred' in args.other_params:
                # stack k predictions into vectors
                agent_num, num_of_predicion_rst, _, _ = gt_trajectory_pair.shape
                assert agent_num == 2, agent_num
                for i in range(agent_num):
                    for j in range(num_of_predicion_rst):
                        pred_vector = np.zeros((time_steps, vectors.shape[1]))
                        pred_vector[:, :2] = gt_trajectory_pair[i, j, :, :2]
                        vectors = np.concatenate([vectors, pred_vector])
                        last_spac_idx = polyline_spans[-1, 1]
                        new_span = np.array([last_spac_idx, last_spac_idx + time_steps]).reshape(1, 2)
                        polyline_spans = np.concatenate([polyline_spans, new_span])
            else:
                # stack k predictions into vectors
                num_of_predicion_rst = influencer_pred_rst.shape[0]
                if 'sub2dec' in args.other_params:
                    num_of_predicion_rst = 1
                for i in range(num_of_predicion_rst):
                    pred_influencer_vector = np.zeros((time_steps, vectors.shape[1]))
                    if 'train_reactor' in args.other_params:
                        # assert time_steps == influencer_pred_rst.shape[1], f'{time_steps} / {influencer_pred_rst.shape} / {gt_trajectory.shape}'
                        pred_influencer_vector[:, :2] = influencer_pred_rst[i, :, :2]
                    vectors = np.concatenate([vectors, pred_influencer_vector])
                    last_spac_idx = polyline_spans[-1, 1]
                    new_span = np.array([last_spac_idx, last_spac_idx + time_steps]).reshape(1, 2)
                    polyline_spans = np.concatenate([polyline_spans, new_span])

    polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]

    if len(lanes) == 0:
        if args.do_eval:
            pass
        else:
            assert False

    stage_one_label = np.argmin([utils.get_dis(lane, gt_trajectory[0, -1, :2]).min() for lane in lanes]) if len(lanes) > 0 else 0

    mapping = {
        'matrix': vectors,
        'polyline_spans': polyline_spans,
        'map_start_polyline_idx': map_start_polyline_idx,
        'labels': labels,
        'labels_is_valid': labels_is_valid,
        'predict_agent_num': predict_agent_num,
        'normalizer': normalizer,
        'goals_2D': goals_2D,
        'polygons': lanes,
        'stage_one_label': stage_one_label,
        'waymo_yaw': waymo_yaw,
        'speed': speed,
        'headings': headings,
        'track_type_int': track_type_int,
        'track_type_string': utils.AgentType.to_string(track_type_int),
        'trajectory_type': trajectory_type,
        'tracks_type': tracks_type,
        'file_name': file_name,
        'instance_id': (scenario_id, objects_id[0]),
        'eval_time': eval_time,

        'yaw_labels': yaw_labels,

        'scenario_id': scenario_id,
        'object_id': objects_id[0],
    }

    # if 'train_reactor' in args.other_params:
    if args.infMLP > 0:
        if 'wscore' in args.other_params:
            mapping['prediction_scores'] = influencer_pred_score[:, np.newaxis]
        mapping['gt_influencer_traj_idx'] = gt_influencer_traj_idx

    mapping.update(mapping_before)

    if eval_time < 80:
        mapping['final_idx'] = eval_time - 1

    # test_vis(trajs, lanes, labels)
    # embed(header='111')
    if args.visualize:
        mapping.update({
            'trajs': trajs,
            'vis_lanes': lanes,
        })

    if 'raster' in args.other_params:
        mapping['image'] = args.image

    final_idx = mapping.get('final_idx', -1)
    mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))

    # Add subgoal info.
    if args.classify_sub_goals:
        if labels.shape[0] > 29:
            mapping['goals_2D_labels_3s'] = np.argmin(utils.get_dis(goals_2D, labels[29]))
        if labels.shape[0] > 49:
            mapping['goals_2D_labels_5s'] = np.argmin(utils.get_dis(goals_2D, labels[49]))

    # Add traffic state info.
    if 'tf_poly' in args.other_params:
        traffic_light_vectors = utils_cython.get_traffic_lights(decoded_example, normalizer, args)
        mapping['traffic_light_vectors'] = traffic_light_vectors

    if args.do_eval:
        mapping.update(mapping_eval)


    return mapping

def test_vis(trajs, lanes, labels):
    import matplotlib.pyplot as plt
    plt.cla()
    fig = plt.figure(0, figsize=(45, 38))

    if True:
        plt.xlim(-100, 100)
        plt.ylim(-30, 100)

    # plt.figure(0, dpi=300)
    cmap = plt.cm.get_cmap('Reds')
    vmin = np.log(0.0001)
    vmin = np.log(0.00001)
    # scores = np.clip(scores.copy(), a_min=vmin, a_max=np.inf)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=np.max(scores)))
    # plt.colorbar(sm)

    for lane in lanes:
        lane = lane[:, :2]
        assert lane.shape == (len(lane), 2), lane.shape
        plt.plot(lane[:, 0], lane[:, 1], linestyle="-", color="black", marker=None,
                 markersize=0,
                 alpha=0.5,
                 linewidth=2,
                 zorder=0)
    linewidth = 5
    
    def draw_his_trajs():
        for i, traj in enumerate(trajs):
            assert isinstance(traj, np.ndarray)
            assert traj.ndim == 2 and traj.shape[1] == 2, traj.shape
            if i == 0:
                traj = np.concatenate([traj, labels], axis=0)
                traj = np.array(traj).reshape([-1])
                t = np.zeros(len(traj))
                t[:len((traj))] = traj
                # t[-2] = labels[0]
                # t[-1] = labels[1]

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
    
    plt.savefig('./test.png', bbox_inches='tight')
    plt.close()



def get_interest_objects(decoded_example, sample_is_valid, gt_future_is_valid, tracks_type, args):
    objects_of_interest = decoded_example['state/objects_of_interest'][sample_is_valid]
    objects_of_interest = objects_of_interest.copy()
    # objects_of_interest = tf.boolean_mask(decoded_example['state/objects_of_interest'], sample_is_valid)
    # objects_of_interest = objects_of_interest.numpy().copy()
    assert objects_of_interest.dtype == np.int64
    assert len(objects_of_interest.shape) == 1

    if objects_of_interest.sum() < 2:
        return None, None

    a, b = [i for i in range(len(objects_of_interest)) if objects_of_interest[i]]

    assert objects_of_interest.sum() == 2, objects_of_interest.sum()

    history_frame_num = 11
    # Ignore examples with invalid last position during training.
    # if not args.do_eval and not (gt_future_is_valid[a, -1] and gt_future_is_valid[b, -1]):
    last_frame_idx = history_frame_num + args.future_frame_num - 1
    if not args.do_eval and not (gt_future_is_valid[a, last_frame_idx] and gt_future_is_valid[b, last_frame_idx]):
        return None, None

    if utils.types_are_ok((tracks_type[a], tracks_type[b]), args):
        pass
    elif utils.types_are_ok((tracks_type[b], tracks_type[a]), args):
        pass
    else:
        return None, None

    if args.inter_agent_types is None or tracks_type[a] == tracks_type[b]:
        if np.random.randint(0, 2) == 0:
            a, b = b, a

    return a, b


def move_interest_objects_forward(decoded_example, sample_is_valid, gt_trajectory, gt_future_is_valid, tracks_type,
                                  objects_id, args,
                                  keep_order=False, order_by_id=0):
    """
    :param order_by_id: 1=smaller first (predict smaller), 2=larger first (predict larger)
    we have reactor at 0th, and influencer at 1st when predict reactor
    this will and should overrule keep order logic
    """

    a, b = get_interest_objects(decoded_example, sample_is_valid, gt_future_is_valid, tracks_type, args)
    
    if a is None:
        return False
    if keep_order:
        if a > b:
            a, b = b, a

    def interactive_swap(tensor, a, b):
        tensor[0], tensor[a] = tensor[a].copy(), tensor[0].copy()
        if b == 0:
            b = a
        tensor[1], tensor[b] = tensor[b].copy(), tensor[1].copy()

    if order_by_id == 1:
        # smaller id first
        if int(objects_id[a]) > int(objects_id[b]):
            a, b = b, a
    elif order_by_id == 2:
        # larger first
        if int(objects_id[a]) < int(objects_id[b]):
            a, b = b, a
    for each in [gt_trajectory, gt_future_is_valid, tracks_type, objects_id]:
        interactive_swap(each, a, b)

    return True
