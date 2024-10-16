


python -m mmlu.run_eval \
    --ntrain 0 \
    --data_dir mmlu/mmlu_data \
    --save_dir results/mmlu/tulu-7B-0shot \
    --model_name_or_path /work/Codes/IniLoRA/scripts/outputs/outputs-v3 \
    --tokenizer_name_or_path /work/Codes/IniLoRA/scripts/outputs/outputs-v3 \
    --eval_batch_size 16 \
    --use_chat_format \
    --load_in_8bit \
    --chat_formatting_function mmlu.templates.create_prompt_with_tulu_chat_format