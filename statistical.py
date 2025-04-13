import torch
from mlp import MLP
import torchvision
from torchvision.transforms import v2
from torch.func import functional_call, grad, vmap
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.linalg import pinv, matrix_rank
from torch.nn.utils import parameters_to_vector
from torch.special import gammainc
from test import test

def compute_fisher(model: nn.Module, x: torch.Tensor, y: torch.Tensor, len_dataset: int) -> dict:
    def compute_loss(params, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        y_hat = functional_call(model, params, (x,))
        loss = F.cross_entropy(y_hat, y)
        return loss

    params = {k: v.detach() for k, v in model.named_parameters()}
    grad_loss = grad(compute_loss)
    all_grads = vmap(grad_loss, in_dims=(None, 0, 0))
    grads = all_grads(params, x, y)

    G = []

    for layer_name in grads.keys():
        print(grads[layer_name][0])
        G.append(grads[layer_name].reshape(len_dataset, -1))


    G = torch.cat(G, dim=1)
    B = (G.T @ G) / len_dataset

    return B
    

def compute_hessian():
    pass

def compute_p(A: torch.Tensor, B: torch.Tensor, A_pinv: torch.Tensor, S: torch.Tensor, theta: torch.Tensor, C: torch.Tensor, len_dataset: int, tolerance: float) -> float:
    Q = (S@C@S.T)

    #Q_pinv = pinv(Q, hermitian=True, rtol=1e-3, atol=1e-6)
    Q_pinv = pinv(Q, hermitian=True, rtol=1e-15, atol=1e-15)

    wald = (theta.T@S.T@Q_pinv@S@theta).view(-1) #* len_dataset

    if (torch.abs(wald) < 0.01):
        wald *= 0

    r = torch.tensor([float(S.shape[0])]).to('cuda')

    p = 1 - gammainc(r/2, wald/2)
    print(f'Wald test: {wald.item()} | p-values: {p.item()}')

    return p

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')



transform = torchvision.transforms.Compose([v2.ToImage(),
                                            v2.ToDtype(torch.uint8, scale=True),
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=(0.5,), std=(0.5,))])


train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                    train=True,
                                                    transform=transform,
                                                    download=False)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                                     train=False,
                                                     transform=transform,
                                                     download=False)

input_dim = train_dataset[0][0].reshape(-1).size(0)
output_dim = len(train_dataset.classes)

model_dict = torch.load('./mnist-mlp.pt')
model = MLP(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(model_dict, strict=False)
model.to(device=device)

#params_vector = parameters_to_vector(model.parameters())




#model.train()

print("Testing the trained model.")
metrics = test(model=model,
               test_dataset=test_dataset,
               device=device)
acc = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1 = metrics['f1']
print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')


trainloader = DataLoader(dataset=train_dataset,
                         batch_size=len(train_dataset),
                         shuffle=False,
                         num_workers=8,
                         pin_memory=True)

x, y = next(iter(trainloader))
x = x.to(device)
y = y.to(device)

B = compute_fisher(model=model,
                   x=x,
                   y=y,
                   len_dataset=len(train_dataset))

#B += 1e-4 * torch.eye(B.shape[0], device=B.device)

# rank = matrix_rank(B)
# if (rank < B.shape[0]):
#     print(f'Rank: {rank}')
#     print("Locally Parameter Redundant")

B_pinv = pinv(B, hermitian=True, rtol=1e-3, atol=1e-5)


#C = B_pinv@B@B_pinv
#print(C)
C = B_pinv
#print(C)
#exit(0)
S_stacked = []

num_params = B.shape[0]

theta = parameters_to_vector(model.parameters()).detach().view(num_params, -1)

num_to_delete = 0

prune_mask = []

for param in range(num_params):
    S = torch.zeros((1, num_params), device=device)
    S[:, param] = 1

    selection_test = S @ B_pinv @ B - S    
    selection_error = selection_test.abs().max().item()
    print(f'Selection error = {selection_error}')

    if (selection_error < 1e-3):
        #print("Parameter could be estimated.")
        p = compute_p(B, B, B_pinv, S, theta, C, len(train_dataset), tolerance=0)
        
        if (p.item() < 0.05):
            prune_mask.append(1)
        else: 
            prune_mask.append(0)
            S_stacked.append(S)
            num_to_delete += 1
    else:
        # Can't reliably estimate this parameter, so keep it
        prune_mask.append(1)
        #print(f"Parameter {param}: Selection condition failed, keeping parameter")

with torch.no_grad():
    current_params_vector = parameters_to_vector(model.parameters())

    prune_mask_tensor = torch.tensor(prune_mask, device=current_params_vector.device, dtype=current_params_vector.dtype)

    pruned_vector = current_params_vector * prune_mask_tensor

    torch.nn.utils.vector_to_parameters(pruned_vector, model.parameters())

print(f'Total parameters: {theta.shape[0]}')
print(f"Number of parameters to delete: {num_to_delete} | Ratio: {num_to_delete/theta.shape[0] * 100:.4f}")


metrics = test(model=model,
               test_dataset=test_dataset,
               device=device)

acc = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1 = metrics['f1']
print(f'Accuracy: {acc * 100:.4f} | Precision {precision* 100:.4f} | Recall {recall* 100:.4f} | F1: {f1* 100:.4f}')





# if S_stacked:
#     print("Checking stacked S to delete multiple parameters.")
#     S_stacked = torch.cat(S_stacked, dim=0).to(device)
#     p_stacked = compute_p(B, B, B_pinv, S_stacked, theta, C, len(train_dataset), 0)
#     print(f'p stacked value: {p_stacked}')

#     if (p_stacked.item() < 0.05):
#         print("Pruning parameters in stacked S.")

#         with torch.no_grad():
#             # 1. Get the current parameters as a flat vector
#             current_params_vector = parameters_to_vector(model.parameters())

#             # 2. Create the mask tensor on the correct device and dtype
#             prune_mask_tensor = torch.tensor(prune_mask, device=current_params_vector.device, dtype=current_params_vector.dtype)

#             # 3. Calculate the pruned vector
#             pruned_vector = current_params_vector * prune_mask_tensor

#             # 4. Copy the pruned vector back into the model's parameters
#             torch.nn.utils.vector_to_parameters(pruned_vector, model.parameters())

#     else: print("Cannot prune")



