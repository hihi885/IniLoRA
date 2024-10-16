


# mnli task
CUDA_VISIBLE_DEVICES=0 python nlu_glue_mnli.py \
    --model_name_or_path FacebookAI/roberta-base \
    --dataset mnli \
    --task mnli \
    --lr 3e-4 \
    --max_length 512 \
    --num_epoch 100 \
    --bs 32  \
    --seed 0 \
    --init_lora_weights sgdSvd \
    --lora_r 8


# other task
CUDA_VISIBLE_DEVICES=0 python nlu_glue_other_tasks.py \
    --model_name_or_path FacebookAI/roberta-base \
    --dataset cola \
    --task cola \
    --lr 3e-4 \
    --max_length 512 \
    --num_epoch 100 \
    --bs 32  \
    --seed 0 \
    --init_lora_weights sgdSvd \
    --lora_r 8