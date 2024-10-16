

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

output_dir=scripts/outputs/outputs-v3

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./deepspeed/stage2_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_lora \
    --init_lora_weights sgdSvd \
    --gradient_checkpointing true \
    --lora_rank 8 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --train_file scripts/raw_train/processed/sharegpt/sharegpt_data.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 3e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --max_train_samples 5 \
    --checkpointing_steps 1000000000 \
    --output_dir ${output_dir} \
    --report_to wandb \
    --with_tracking \
    --logging_steps 5
