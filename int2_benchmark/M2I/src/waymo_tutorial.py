import numpy as np
import tensorflow as tf
import os
import pickle
from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from IPython import embed

def _parse(data, hdmap_dir):

    decoded_example = data
    # Combining agent information with high-definition map information.
    map_id = decoded_example['scenario/map_id']
    map_info_path = os.path.join(hdmap_dir, map_id + '.pickle')

    with open(map_info_path, 'rb+') as f:
        map_info = pickle.load(f)

    decoded_example['roadgraph_samples/dir'] = map_info['roadgraph_samples/dir']
    decoded_example['roadgraph_samples/xyz'] = map_info['roadgraph_samples/xyz']
    decoded_example['roadgraph_samples/id'] = map_info['roadgraph_samples/id']
    decoded_example['roadgraph_samples/type'] = map_info['roadgraph_samples/type']
    decoded_example['roadgraph_samples/valid'] = map_info['roadgraph_samples/valid']
    
    past_states = np.concatenate([
        decoded_example['state/past/x'][:, :, np.newaxis],
        decoded_example['state/past/y'][:, :, np.newaxis],
        decoded_example['state/past/length'][:, :, np.newaxis],
        decoded_example['state/past/width'][:, :, np.newaxis],
        decoded_example['state/past/bbox_yaw'][:, :, np.newaxis],
        decoded_example['state/past/velocity_x'][:, :, np.newaxis],
        decoded_example['state/past/velocity_y'][:, :, np.newaxis]
    ], axis=2)

    cur_states = np.concatenate([
        decoded_example['state/current/x'][:, :, np.newaxis],
        decoded_example['state/current/y'][:, :, np.newaxis],
        decoded_example['state/current/length'][:, :, np.newaxis],
        decoded_example['state/current/width'][:, :, np.newaxis],
        decoded_example['state/current/bbox_yaw'][:, :, np.newaxis],
        decoded_example['state/current/velocity_x'][:, :, np.newaxis],
        decoded_example['state/current/velocity_y'][:, :, np.newaxis]
    ], axis=2)

    # print(f'======================= past_states: {past_states.shape}, cur_states: {cur_states.shape}')
    input_states = np.concatenate([past_states, cur_states], axis=1)[:, :, :2]

    future_states = np.concatenate([
        decoded_example['state/future/x'][:, :, np.newaxis],
        decoded_example['state/future/y'][:, :, np.newaxis],
        decoded_example['state/future/length'][:, :, np.newaxis],
        decoded_example['state/future/width'][:, :, np.newaxis],
        decoded_example['state/future/bbox_yaw'][:, :, np.newaxis],
        decoded_example['state/future/velocity_x'][:, :, np.newaxis],
        decoded_example['state/future/velocity_y'][:, :, np.newaxis]
    ], axis=2)

    gt_future_states = np.concatenate([past_states, cur_states, future_states], axis=1)

    past_is_valid = decoded_example['state/past/valid'] > 0
    current_is_valid = decoded_example['state/current/valid'] > 0
    future_is_valid = decoded_example['state/future/valid'] > 0

    gt_future_is_valid = np.concatenate([past_is_valid, current_is_valid, future_is_valid], axis=1)

    sample_is_valid = np.any(np.concatenate([past_is_valid, current_is_valid], axis=1), axis=1)
    
    inputs = {
        'input_states': input_states,
        'gt_future_states': gt_future_states,
        'gt_future_is_valid': gt_future_is_valid,
        'object_type': decoded_example['state/type'],
        'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
        'interactive_tracks_to_predict': decoded_example['state/objects_of_interest'] > 0,
        'sample_is_valid': sample_is_valid,
    }

    return inputs, decoded_example


def _default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    track_future_samples: 80
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
        measurement_step: 5
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
    }
    step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
    }
    step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
    }
    max_predictions: 6
    """
    text_format.Parse(config_text, config)
    return config


class MotionMetrics:
    """Wrapper for motion metrics computation."""

    def __init__(self, config, is_short=False):
        # super().__init__()
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []
        self._scenario_id = []
        self._object_id = []
        self._metrics_config = config
        self.is_short = is_short
        self.not_compute = False

        self.args = None

    def reset_state(self):
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []
        self._scenario_id = []
        self._object_id = []

    def update_state(self, prediction_trajectory, prediction_score,
                     ground_truth_trajectory, ground_truth_is_valid, object_type, scenario_id, object_id):
        if self.is_short:
            interval = (
                    self._metrics_config.track_steps_per_second //
                    self._metrics_config.prediction_steps_per_second)
            assert len(prediction_trajectory.shape) == 4, prediction_trajectory.shape
            
            if not isinstance(prediction_trajectory, np.ndarray):
                prediction_trajectory = prediction_trajectory.numpy()
            if prediction_trajectory.shape[2] == self.args.future_frame_num:
                prediction_trajectory = prediction_trajectory[:, :, (interval - 1)::interval, :].copy()
            else:
                assert prediction_trajectory.shape[2] == 16
            ground_truth_trajectory = None
            ground_truth_is_valid = None

        self._prediction_trajectory.append(prediction_trajectory)
        self._prediction_score.append(prediction_score)
        self._ground_truth_trajectory.append(ground_truth_trajectory)
        self._ground_truth_is_valid.append(ground_truth_is_valid)
        self._object_type.append(object_type)
        self._scenario_id.append(scenario_id)
        self._object_id.append(object_id)

    def get_all(self):
        return (
            self._prediction_trajectory,
            self._prediction_score,
            self._ground_truth_trajectory,
            self._ground_truth_is_valid,
            self._object_type,
            self._scenario_id,
            self._object_id,
        )

    def result(self):
        # [batch_size, steps, 2].
        if self.is_short or self.not_compute:
            return None
        if len(self._prediction_trajectory) == 0:
            return None
        prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
        # [batch_size].
        prediction_score = tf.concat(self._prediction_score, 0)
        # [batch_size, gt_steps, 7].
        ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
        # [batch_size, gt_steps].
        ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
        # [batch_size].
        object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

        # We are predicting more steps than needed by the eval code. Subsample.
        interval = (
                self._metrics_config.track_steps_per_second //
                self._metrics_config.prediction_steps_per_second)
        
        # [batch_size, top_k, num_agents_per_joint_prediction, pred_steps, 2].
        if len(prediction_trajectory.shape) == 5:
            prediction_trajectory = prediction_trajectory[:, :, :, (interval - 1)::interval, :]
        else:
            assert len(prediction_trajectory.shape) == 4, prediction_trajectory.shape
            prediction_trajectory = prediction_trajectory[:, :, tf.newaxis, (interval - 1)::interval, :]

        # Prepare these into shapes expected by the metrics computation.
        #
        # num_agents_per_joint_prediction is also 1 here.
        # [batch_size, top_k].
        assert len(prediction_score.shape) == 2
        prediction_score = prediction_score[:, :]
        # [batch_size, num_agents_per_joint_prediction, gt_steps, 7].
        if len(ground_truth_trajectory.shape) == 4:
            pass
        else:
            ground_truth_trajectory = ground_truth_trajectory[:, tf.newaxis]
        # # SQ: change to hard checking, adding a new axis at the end to fit target dimension does not make sense ->>>>>
        # assert len(ground_truth_trajectory.shape) == 4, ground_truth_trajectory.shape

        # [batch_size, num_agents_per_joint_prediction, gt_steps].
        if len(ground_truth_is_valid.shape) == 3:
            pass
        else:
            ground_truth_is_valid = ground_truth_is_valid[:, tf.newaxis]
        # [batch_size, num_agents_per_joint_prediction].
        if len(object_type.shape) == 2:
            pass
        else:
            object_type = object_type[:, tf.newaxis]

        return py_metrics_ops.motion_metrics(
            config=self._metrics_config.SerializeToString(),
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            object_type=object_type)


metrics_config = _default_metrics_config()
motion_metrics = MotionMetrics(metrics_config)
metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)
