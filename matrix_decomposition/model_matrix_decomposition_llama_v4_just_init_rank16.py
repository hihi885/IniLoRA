import concurrent.futures
import time
import os

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import transformers
from transformers import set_seed
import torch


def sgd_svd(name, weight, device):
    global std
    file_path = f'{save_dir}/{name}.pt'
    if os.path.exists(file_path):
        return

    print(f'std: {std}')
    U = Variable(torch.normal(0, std, size=(4096, K)).to(device), requires_grad=True)
    P = Variable(torch.normal(0, std, size=(4096, K)).to(device), requires_grad=True)

    # v1
    # U = Variable(torch.normal(0, 0.5, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 0.5, size=(4096, K)).to(device), requires_grad=True)

    # v2
    # U = Variable(torch.normal(0, 0.1, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 0.1, size=(4096, K)).to(device), requires_grad=True)

    # v3
    # U = Variable(torch.normal(0, 0.01, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 0.01, size=(4096, K)).to(device), requires_grad=True)

    # v4
    # U = Variable(torch.normal(0, 0.001, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 0.001, size=(4096, K)).to(device), requires_grad=True)

    # v5
    # U = Variable(torch.normal(0, 0.0001, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 0.0001, size=(4096, K)).to(device), requires_grad=True)

    # v6
    # U = Variable(torch.normal(0, 0.0, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 0.0, size=(4096, K)).to(device), requires_grad=True)

    A, B = U.cpu().detach().clone(), P.cpu().detach().clone().T
    torch.save({'A': A, 'B': B}, file_path)


if __name__ == '__main__':
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "/work/Codes/models/llama2-7b-hf-meta",
    )
    K = 16
    # for version, std in [['v1', 0.5], ['v2', 0.1],['v3', 0.01],['v4', 0.001],['v5', 0.0001]]:
    for version, std in [['v7', 0.75], ['v8', 1], ['v9', 1.5]]:
        for seed, postfix in [[44, ''], [1024, '_1'], [2048, '_2']]:
            set_seed(seed)

            save_dir = f'./var_llama2-7b/llama-7b-hf-decomposition-rank-{K}-just-init-weight-{version}{postfix}'
            print(f'save dir: {save_dir}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            tasks = []
            elements = ['v_proj', 'q_proj']
            for name, para in model.named_parameters():
                if any([e in name for e in elements]):
                    tasks.append([name, para.data])

            for i in range(4):
                start = i * int(len(tasks) / 4)
                end = (i + 1) * int(len(tasks) / 4)
                for j in range(start, end):
                    tasks[j].append(torch.device(f'cpu'))
            for e in tasks:
                assert len(e) == 3

            tasks = tasks[:]
            POOL_SIZE = len(tasks)
            assert POOL_SIZE == len(tasks)
            with concurrent.futures.ThreadPoolExecutor(max_workers=POOL_SIZE) as executor:
                futures = [executor.submit(sgd_svd, *t) for t in tasks]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
