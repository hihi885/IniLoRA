import concurrent.futures
import os

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import transformers
from transformers import set_seed
import torch

std_list = []

def sgd_svd(name, weight, device):
    file_path = f'{save_dir}/{name}.pt'
    if os.path.exists(file_path):
        return

    ratings_matrix = weight.data.clone().detach()

    print(f'std: {ratings_matrix.std()}')
    std_list.append(ratings_matrix.std())

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

    # v7
    # U = Variable(torch.normal(0, 0.75, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 0.75, size=(4096, K)).to(device), requires_grad=True)

    # v8
    # U = Variable(torch.normal(0, 1, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 1, size=(4096, K)).to(device), requires_grad=True)

    # # v9
    # U = Variable(torch.normal(0, 1.5, size=(4096, K)).to(device), requires_grad=True)
    # P = Variable(torch.normal(0, 1.5, size=(4096, K)).to(device), requires_grad=True)

    # v10
    U = Variable(torch.normal(0, 0.25, size=(4096, K)).to(device), requires_grad=True)
    P = Variable(torch.normal(0, 0.25, size=(4096, K)).to(device), requires_grad=True)

    A, B = U.cpu().detach().clone(), P.cpu().detach().clone().T
    torch.save({'A': A, 'B': B}, file_path)


if __name__ == '__main__':
    # set_seed(44)
    # set_seed(1024)
    set_seed(2048)

    K = 8

    save_dir = f'./var_llama2-7b/llama-7b-hf-decomposition-rank-{K}-just-init-weight-v10_2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "/work/Codes/models/llama2-7b-hf-meta",
    )

    tasks = []
    elements = ['v_proj', 'q_proj']
    for name, para in model.named_parameters():
        if any([e in name for e in elements]):
            print(name)
            print(para.shape)
            tasks.append([name, para.data])

    for i in range(4):
        start = i * int(len(tasks) / 4)
        end = (i + 1) * int(len(tasks) / 4)
        for j in range(start, end):
            tasks[j].append(torch.device(f'cuda:{0}'))
    for e in tasks:
        assert len(e) == 3

    tasks = tasks[:]
    POOL_SIZE = len(tasks)
    assert POOL_SIZE == len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=POOL_SIZE) as executor:
        futures = [executor.submit(sgd_svd, *t) for t in tasks]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()

    print('mean std: ')
    print(sum(std_list) / len(std_list))
