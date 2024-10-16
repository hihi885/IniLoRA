#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
root_path=/work/Codes/IniLoRA/scripts
version=v1

python inference/MATH_inference.py --batch_size 64 --model ${root_path}/outputs/outputs_${version}/

