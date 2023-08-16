# train marginal
## if you want to debug, please remove the last line and add --debug_mode

## train rush_hour with all type
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

# --debug_mode



## train non_rush_hour with all type
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