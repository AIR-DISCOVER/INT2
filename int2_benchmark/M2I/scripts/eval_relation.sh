# Eval marginal
## eval rush_hour with all type
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

# --debug_mode



## eval rush_hour with all type
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