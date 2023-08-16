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


**Step 5:** Runing Model.
```
cd ../  # ~/int2_benchmark/M2I/
```

## Train Relation

**Step 1: Train Relation** 

Train rush_hour with all type
```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/rush_hour/relation; \
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
OUTPUT_DIR=../output/non_rush_hour/relation; \
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

<hr>

**Step 2: Train Marginal** 

Train rush_hour with all type
```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/rush_hour/marginal; \
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
OUTPUT_DIR=../output/non_rush_hour/marginal; \
HDMAP_DIR='../../../int2_dataset/m2i_format/hdmap'
AGENT_TYPE=vehicle; \
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m src.run --do_train --waymo \
--data_txt ${DATA_TXT} \
--hdmap_dir ${HDMAP_DIR} --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 128 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--distributed_training 4  --master_port 63242
```

<hr>

**Step 3: Train Conditional** 

Train rush_hour with all type
```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/rush_hour/marginal; \
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
OUTPUT_DIR=../output/non_rush_hour/marginal; \
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

<hr>


**Addition**
if you want to debug, please remove the last line and add --debug_mode
for example, when you want to debug train relation with rush_hour dataset, you can running the script:

```
DATA_TXT=./domain/rush_hour_train.txt; \
OUTPUT_DIR=../output/rush_hour/relation; \
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