# Eval conditional
## eval rush_hour with all type
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




## eval non_rush_hour with all type
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