# CUDA_VISIBLE_DEVICES=$1 python3 qwen_inference_from_dump.py --preprocessed_dir ./embeddings \
#                                 --output_file ./results/full_kv.json --model_path Qwen/${2}
MODEL_PATH=Qwen/Qwen2.5-VL-${2}B-Instruct
BLOCK_SIZE=$3
COMPRESS_FRAME_NUM=$4
COMPRESSION_METHOD=$5
EXP_TAG=${2}B_${BLOCK_SIZE}_${COMPRESS_FRAME_NUM}_${COMPRESSION_METHOD}

CUDA_VISIBLE_DEVICES=$1 python3 qwen_inference_ovu.py \
    --dataset $6 \
    --output_dir results/ovu \
    --exp_tag $EXP_TAG \
    --use_block_processing \
    --block_size $BLOCK_SIZE \
    --compress_frame_num $COMPRESS_FRAME_NUM \
    --model_path $MODEL_PATH \
    --load_dumped --adaptive_pooling \
    --compression_method $COMPRESSION_METHOD
