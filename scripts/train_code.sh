#!/bin/bash


root_path=/work/Codes/IniLoRA/scripts
version=v2
deepspeed_config=../deepspeed/ds_stage2_v2.json
deepspeed --master_port 29506 --include localhost:0 train_code.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path /work/Codes/IniLoRA/data/code_alpaca_20k.json \
    --output_dir ${root_path}/outputs/outputs_${version}/ \
    --init_lora_weights sgdSvd \
    --report_to tensorboard \
    --logging_dir ${root_path}/logs/${version} \
    --seed 11 \
    --query "query" \
    --response "response" \
    --model_max_length 512 \
    --merge_and_save True \
    --lora_r 8 \
    --data_length 20000 \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed ${root_path}/${deepspeed_config}

