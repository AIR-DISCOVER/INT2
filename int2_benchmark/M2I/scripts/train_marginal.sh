# train marginal

# ======================== vv peak

DATA_DIR=/DATA_EDS/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
DATA_TXT=/DATA_EDS/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-vehicle/peak_train_data.txt; \

OUTPUT_DIR=/DATA_EDS/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-vehicle/peak/marginal; \

CUDA_VISIBLE_DEVICES=2,3 python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 128 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type vehicle \
--distributed_training 2


# --distributed_training 4 --master_port 63236


# --debug_mode

# --distributed_training 5 --master_port 63236
# --distributed_training 4 --master_port 63236



# ======================== vv trough

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-vehicle/trough_train_data.txt; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-vehicle/trough/marginal; \

python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type vehicle \
--distributed_training 8 --master_port 63212




# ======================== vc peak

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-cyclist/peak_train_data.txt; \
OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-cyclist/peak/marginal; \

python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type vehicle \
--distributed_training 4 --master_port 63212



# ======================== vc trough

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-cyclist; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-cyclist/trough_train_data.txt; \
OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-cyclist/trough/marginal; \

python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 128 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type vehicle \
--distributed_training 3 --master_port 63212







# ======================== vp peak

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \
DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-pedestrian/peak_valid_data.txt; \
OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-pedestrian/peak/marginal; \

python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type vehicle \
--distributed_training 4




# ======================== vp trough

DATA_DIR=/home/DISCOVER/yanzj/workspace/code/INT2/dataset/INT2/vehicle-pedestrian; \

DATA_TXT=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-pedestrian/trough_train_data.txt; \

OUTPUT_DIR=/home/DISCOVER/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-pedestrian/trough/marginal; \

python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 --sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--future_frame_num 80 --agent_type vehicle \
--distributed_training 4 --master_port 63212