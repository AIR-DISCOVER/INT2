# Eval marginal
## eval rush_hour with all type
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


# --debug_mode


## eval non_rush_hour with all type
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