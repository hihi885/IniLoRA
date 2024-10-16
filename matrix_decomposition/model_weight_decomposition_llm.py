import concurrent.futures
import argparse
import time
import os

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import transformers
import torch

available_models = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf',
                    'meta-llama/Meta-Llama-3-8B', 'google/gemma-7b']

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=available_models, help='model for weight decomposition', required=True)
args = parser.parse_args()


def sgd_svd(name, weight, module):
    file_path = f'{save_dir}/{name}.pt'
    if os.path.exists(file_path):
        return

    device = 'cuda:0'
    ratings_matrix = weight.data.clone().detach()
    ratings_matrix = ratings_matrix.to(device)

    init_std = init_global_para[module]
    U = Variable(torch.normal(0, init_std, size=(ratings_matrix.shape[0], K)).to(device), requires_grad=True)
    P = Variable(torch.normal(0, init_std, size=(ratings_matrix.shape[1], K)).to(device), requires_grad=True)

    optimizer = torch.optim.Adam([U, P], lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)
    early_stop = False
    loss_interval = 0.000005
    patience = 5
    patience_counter = 0
    last_loss = None
    loss_list = []

    print(f'Start sgd_svd, total Iteration: {iter_num}')
    for i in range(iter_num):
        optimizer.zero_grad()

        predicted_ratings = torch.mm(U, P.t())

        loss = ((predicted_ratings - ratings_matrix) ** 2).mean()  # + lambda_reg * (U.pow(2).sum() + P.pow(2).sum())
        loss_list.append(loss.item())

        loss.backward()
        scheduler.step()
        optimizer.step()

        if i % 50 == 0:
            print(f'Iteration {i}, Loss: {loss.item()}')

            if early_stop:
                if i == 0:
                    last_loss = loss.item()
                    continue
                if last_loss - loss.item() > loss_interval:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'break at Iteration: {i}')
                        break
                last_loss = loss.item()

    A, B = U.cpu().detach().clone(), P.cpu().detach().clone().T
    torch.save({'A': A, 'B': B}, file_path)

    loss_file_path = f'{save_dir}/{name}.loss.pt'
    torch.save({'loss': loss_list}, loss_file_path)


def calcu_global_parameters(model):
    global_para = {'v_proj': [], 'q_proj': []}
    for element in elements:
        std_list = []
        for name, para in model.named_parameters():
            if element in name:
                std_list.append(para.data.std())
        global_para[element] = round((sum(std_list) / len(std_list)).item(), 3)
    return global_para


if __name__ == '__main__':
    K = 8
    lambda_reg = 0.0002
    learning_rate = 5e-4
    iter_num = 20000
    elements = ['v_proj', 'q_proj']
    model = args.model

    save_dir = f'./init_weights/{os.path.basename(model)}/rank-{K}-iterNum-{iter_num}-lr-{learning_rate}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'save dir: {save_dir}')

    model = transformers.AutoModelForCausalLM.from_pretrained(model)

    init_global_para = calcu_global_parameters(model)
    print(f'init global parameter: {init_global_para}')

    tasks = []

    for name, para in model.named_parameters():
        if any([e in name for e in elements]):
            module_name = 'q_proj' if 'q_proj' in name else 'v_proj'
            tasks.append([name, para.data, module_name])

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = [executor.submit(sgd_svd, *t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
