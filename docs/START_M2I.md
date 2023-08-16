**Step 1:** download INT2 Dataset refer this <a href="DOWNLOADING.md">link</a>. If you have already downloaded, just skip this step

**Step 2:** Create an python environment for M2I
```shell
conda activate int2
pip install -r requirements.txt
```

**Step 3:** preprocess INT2 Dataset to M2I needs format.

```
# Edit python data_format_preprocess_m2i.py, modify the path in the main function to the path you want, default does not need to be modified. The process takes about 12 hours.

python data_format_preprocess_m2i.py
```

**Step 4:** Runing configuration.
```
cp dataset_int2.py M2I/src
cd M2I/src
cython -a utils_cython.pyx && python setup.py build_ext --inplace

# split dataset to rush-hour or non-rush-hour
python split_dataset.py

# Then, you will found four TXT files are generated in int2_benchmark/M2I/src/domain
```


## Train Model

```
cd ../  # ~/int2_benchmark/M2I/
```


**Step 1: Train Relation** 

Train rush_hour with all type
```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/train/rush_hour/relation; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 216 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 4 --master_port 63239
```

Train non_rush_hour with all type

```
DATA_TXT=./domain/non_rush_hour_train.txt; \
OUTPUT_DIR=../output/train/non_rush_hour/relation; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 216 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 4 --master_port 63240
```


**Step 2: Train Marginal** 

Train rush_hour with all type
```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/train/rush_hour/marginal; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train --waymo \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 128 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--distributed_training 4  --master_port 63241
```

Train non_rush_hour with all type

```
DATA_TXT=./domain/non_rush_hour_train.txt; \
OUTPUT_DIR=../output/train/non_rush_hour/marginal; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train --waymo \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 128 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--distributed_training 4  --master_port 63242
```



**Step 3: Train Conditional** 

Train rush_hour with all type
```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/train/rush_hour/marginal; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 4
```

Train non_rush_hour with all type

```
DATA_TXT=./domain/non_rush_hour_train.txt; \
OUTPUT_DIR=../output/train/non_rush_hour/marginal; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 4
```




**Addition**
if you want to debug, please remove the last line and add `--debug_mode`
for example, when you want to debug train relation with rush_hour dataset, you can running the script:

```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/train/rush_hour/relation; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 216 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--debug_mode
```
When you meet CUDA OUT Of Memory problem, you can decrease train_batch_size.




## Eval Model

**Step 1: Eval Relation** 

If you want to eval in-domain results, please fellow the bottom script and change `train_relation_checkpoint_path` to your train relation result model (default in `int2_benchmark/M2I/output/train/*`). If you want to eval cross-domain results, you can change `train_relation_checkpoint_path` to other domain validation data.

Eval rush_hour with all type

```
OUTPUT_DIR=../output/eval/rush_hour/relation; \
DATA_TXT=./domain/rush_hour_val.txt; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --waymo \
--data_txt ${DATA_TXT} \
--config relation.yaml --hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} \
--future_frame_num 80  --agent_type ${AGENT_TYPE}\
--nms_threshold 7.2 -e \
--validation_model 10 --relation_pred_threshold 0.9 \
--model_recover_path train_relation_checkpoint_path
--distributed_training 4 --master_port 63251
```

Eval non_rush_hour with all type

```
OUTPUT_DIR=../output/eval/non_rush_hour/relation; \
DATA_TXT=./domain/non_rush_hour_val.txt; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --waymo \
--data_txt ${DATA_TXT} \
--config relation.yaml --hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} \
--future_frame_num 80  --agent_type ${AGENT_TYPE}\
--nms_threshold 7.2 -e \
--validation_model 10 --relation_pred_threshold 0.9 \
--model_recover_path train_relation_checkpoint_path
--distributed_training 4 --master_port 63252
```


**Step 2: Eval Marginal** 

Modify ```eval_relation_result_path``` to be the path of the eval relation result path and modify ```train_marginal_checkpoint_path``` to be the path of the train marginal result path.

Eval rush_hour with all type

```
OUTPUT_DIR=../output/eval/rush_hour/marginal; \
DATA_TXT=./domain/rush_hour_val.txt; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train --waymo \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hdmap_dir ${HDMAP_DIR} --hidden_size 128 --train_batch_size 256 \
--sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster train_pair_interest save_rst \
--future_frame_num 80 --agent_type vehicle -e --nms 7.2 \
--eval_exp_path eval_relation_result_path \
--model_recover_path train_marginal_checkpoint_path \
--distributed_training 4 --master_port 63255
```

Eval non_rush_hour with all type

```
OUTPUT_DIR=../output/eval/non_rush_hour/marginal; \
DATA_TXT=./domain/non_rush_hour_val.txt; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train --waymo \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hdmap_dir ${HDMAP_DIR} --hidden_size 128 --train_batch_size 256 \
--sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster train_pair_interest save_rst \
--future_frame_num 80 --agent_type vehicle -e --nms 7.2 \
--eval_exp_path eval_relation_result_path \
--model_recover_path train_marginal_checkpoint_path \
--distributed_training 4 --master_port 63256
```


**Step 3: Eval Conditional** 

Modify ```eval_relation_result_path``` to be the path of the eval relation result path,
modify ```eval_marginal_result_path``` to be the path of the eval marginal result path,
modify ```train_conditional_checkpoint_path``` to be the path of the train conditional result path.

Eval rush_hour with all type

```
OUTPUT_DIR=../output/eval/rush_hour/conditional; \
RESULT_EXPORT_PATH=../output/eval/rush_hour/conditional; \
DATA_TXT=./domain/rush_hour_val.txt; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --waymo \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hdmap_dir ${HDMAP_DIR} --config conditional_pred.yaml \
--future_frame_num 80 \
-e --eval_rst_saving_number 0 \
--eval_exp_path RESULT_EXPORT_PATH \
--relation_pred_file_path eval_relation_result_path \
--influencer_pred_file_path eval_marginal_result_path \
--model_recover_path train_conditional_checkpoint_path \
--distributed_training 4 --master_port 63265
```

Eval non_rush_hour with all type

```
OUTPUT_DIR=../output/eval/non_rush_hour/conditional; \
RESULT_EXPORT_PATH=../output/eval/non_rush_hour/conditional; \
DATA_TXT=./domain/non_rush_hour_val.txt; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --waymo \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hdmap_dir ${HDMAP_DIR} --config conditional_pred.yaml \
--future_frame_num 80 \
-e --eval_rst_saving_number 0 \
--eval_exp_path RESULT_EXPORT_PATH \
--relation_pred_file_path eval_relation_result_path \
--influencer_pred_file_path eval_marginal_result_path \
--model_recover_path train_conditional_checkpoint_path \
--distributed_training 4 --master_port 63266
```


More detailed results can be found <a href="../int2_benchmark/M2I/scripts">here</a>.