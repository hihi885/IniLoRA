


output_dir=/work/Codes/IniLoRA/scripts/outputs/outputs_v2

CUDA_VISIBLE_DEVICES=0 python -m codex_humaneval.run_eval \
    --data_file ./codex_humaneval/HumanEval.jsonl \
    --model_name_or_path ${output_dir} \
    --tokenizer_name_or_path ${output_dir} \
    --use_slow_tokenizer \
    --save_dir ${output_dir}/codex_eval_results/pass1 \
    --eval_pass_at_ks 1 5 10 20 \
    --temperature 0.1 \
    --unbiased_sampling_size_n 20 \
    --eval_batch_size 4 \
    --use_vllm