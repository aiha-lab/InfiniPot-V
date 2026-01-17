MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
BLOCK_SIZE=16
COMPRESS_FRAME_NUM=12
COMPRESSION_METHOD=infinipot-v
EXP_TAG=7B_${BLOCK_SIZE}_${COMPRESS_FRAME_NUM}_${COMPRESSION_METHOD}
MAX_FRAMES_NUM=768

CUDA_VISIBLE_DEVICES=$1 python3 qwen_inference_ovu.py \
    --dataset sample \
    --output_dir results/sample \
    --exp_tag $EXP_TAG \
    --use_block_processing \
    --block_size $BLOCK_SIZE \
    --compress_frame_num $COMPRESS_FRAME_NUM \
    --model_path $MODEL_PATH \
    --compression_method $COMPRESSION_METHOD \
    --max_frames_num $MAX_FRAMES_NUM --verbose