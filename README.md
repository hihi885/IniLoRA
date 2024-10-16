# IniLoRA


## IniLoRA Method

### Weight approximation experiments

- The weights of the q and v modules of llama2-7b are approximated
```shell
cd matrix_decomposition && python model_weight_decomposition_llm.py --model meta-llama/Llama-2-7b-hf
```

- or: The weights of the q and v modules of roberta are approximated
```shell
cd matrix_decomposition && python model_weight_decomposition_roberta.py --model FacebookAI/roberta-base
```

- The results of the weight approximation are saved in the directory `/work/Codes/IniLoRA/matrix_decomposition/init_weights/Llama-2-7b-hf/rank-8-iterNum-20000-lr-0.0005/`, referred to as `weight_init_path`.
  
- Modify the `/work/Codes/IniLoRA/peft/tuners/lora/layer.py` file to set the value of the `root_path` variable in the `sgd_svd` function to `weight_init_path`

### Training and Evaluation

#### Performance on GSM8K/MATH

- Download the training set `MetaMathQA-395K.json` from `https://huggingface.co/datasets/meta-math/MetaMathQA/tree/main` and place it in the `data` folder
  
- Execute `bash scripts/train_gsm8k_math.sh` to start training.

- Run `bash scripts/test_gsm8k.sh` to evaluate on the GSM8K test set.

- Run `bash scripts/test_math.sh` to evaluate on the MATH test set.


#### Performance on HumanEval

- Download the training set `code_alpaca_20k.json` from `https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/tree/main` and place it in the `data` folder
  
- Run `bash scripts/train_code.sh` to start training.

- Run `cd scripts && bash test_humaneval.sh` to evaluate on the HumanEval benchmark.

#### Performance on MMLU

- Data processing `cd scripts && prepare_data.sh`

- Run `bash scripts/train_mmlu.sh` to start training

- Run `cd scripts && bash test_mmlu.sh` to evaluate on the MMLU benchmark.


#### Performance on GLUE

- Run `bash scripts/train_glue_tasks.sh` to perform training and evaluation.

## IniLoRA-alpha Method

- Execute `bash scripts/train_with_IniLoRA_alpha.sh` to start training.

## IniLoRA-beta Method

- Execute `bash scripts/train_with_IniLoRA_beta.sh` to start training.