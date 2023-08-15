# train conditional


# ======================== vv peak
DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-vehicle/peak_train_data.txt; \
RELATION_GT_DIR=/home/DISCOVER//yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-vehicle/peak/conditional; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 8


--debug_mode
# --distributed_training 4 --master_port 63262



# --distributed_training 1 --master_port 63262
# --debug_mode --master_port 63262




# ======================== vv trough
DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-vehicle/trough_train_data.txt; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-vehicle/trough/conditional; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 8 --master_port 64555


--debug_mode 




# ======================== vc peak
DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-cyclist/peak_train_data.txt; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-cyclist/peak/conditional; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 160 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 4



# ======================== vc trough
DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-cyclist/trough_train_data.txt; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-cyclist/trough/conditional; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 128 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 4





# ======================== vp peak
OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-pedestrian/peak/conditional; \
DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-pedestrian/peak_train_data.txt; \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \
python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 160 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 4 --master_port 64663








# ======================== vp trough
DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \

DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-pedestrian/trough_train_data.txt; \

AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-pedestrian/trough/conditional; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 300 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type ${AGENT_TYPE} \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj ${PAIR_TYPE} raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 6