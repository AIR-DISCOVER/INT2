OUTPUT_DIR=./vis/vv_trough_trough_marginal_svg; \
DATA_DIR=/DATA_EDS/yanzj/workspace/code/INT2/dataset/INT2/vehicle-vehicle; \
DATA_TXT=/DATA_EDS/yanzj/workspace/code/INT2_Benchmark/dataset/vehicle-vehicle/trough_valid_data.txt; \

CUDA_VISIBLE_DEVICES=1 python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--data_txt ${DATA_TXT} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 256 \
--sub_graph_batch_size 4096 --core_num 16 \
--other_params l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster train_pair_interest save_rst \
--debug_mode --future_frame_num 80 --agent_type vehicle -e --nms 7.2 --eval_exp_path validation_interactive_v_rdensetnt_full \
--model_recover_path /DATA_EDS/yanzj/workspace/code/INT2_Benchmark/output/iccv/vehicle-vehicle/trough/marginal/2023-03-02-21-45-13/model_save/model.25.bin \
--visualize


