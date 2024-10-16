import argparse

print('Parsing args')

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
parser.add_argument("--dataset", type=str, default="mrpc")
parser.add_argument("--task", type=str, default="mrpc")
parser.add_argument("--bs", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warm_step", type=float, default=0.06)
parser.add_argument("--train_ratio", type=float, default=1)
parser.add_argument("--width", type=float, default=200.)
parser.add_argument("--set_bias", action= "store_true")
parser.add_argument("--seed", type=int, default=00000)
parser.add_argument("--entry_seed", type=int, default=2024)

parser.add_argument("--init_lora_weights", type=str, default="lora")
parser.add_argument("--lora_r", type=int, default=8)
args = parser.parse_args()

def get_args():
    return parser.parse_args()