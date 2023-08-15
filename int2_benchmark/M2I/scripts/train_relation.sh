# train relation

# vv peak

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-vehicle/peak_train_data.txt; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-vehicle/peak/relation; \
RELATION_GT_DIR=/home/DISCOVER//yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 512 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation ${PAIR_TYPE} \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 4

# --debug_mode --master_port 63235

# --distributed_training 2  --master_port 63235





# vv trough

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-vehicle/trough_train_data.txt; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-vehicle/trough/relation; \
RELATION_GT_DIR=/home/DISCOVER//yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation ${PAIR_TYPE} \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 3 --master_port 63235








# vc peak

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-cyclist/peak_train_data.txt; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-cyclist/peak/relation; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 128 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation ${PAIR_TYPE} \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 2 --master_port 63236






# vc trough

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-cyclist/trough_train_data.txt; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-cyclist/trough/relation; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 512 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation ${PAIR_TYPE} \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 4 --master_port 63239
















# vp peak

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-pedestrian/peak_train_data.txt; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-pedestrian/peak/relation; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation ${PAIR_TYPE} \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 4






# vp trough

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \
RELATION_GT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-pedestrian/trough_train_data.txt; \
OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-pedestrian/trough/relation; \
AGENT_TYPE=vehicle; \
PAIR_TYPE=pair_vv; \

python -m src.run --do_train --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type ${AGENT_TYPE} \
--other_params train_relation ${PAIR_TYPE} \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 4 --master_port 63239