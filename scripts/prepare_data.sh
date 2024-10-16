


echo "Downloading ShareGPT dataset..."
wget -P raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
wget -P raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json

echo "Splitting the ShareGPT dataset with 2048 max tokens per conversation..."
python split_sharegpt_conversations.py \
    --in-files raw_train/sharegpt/sg_90k_part1_html_cleaned.json raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048.json \
    --model-name-or-path oobabooga/llama-tokenizer \
    --max-length 2048

echo "Processing datasets..."
python reformat_datasets.py --raw_data_dir raw_train/ --output_dir raw_train/processed/ --dataset sharegpt