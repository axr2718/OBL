import torch.nn as nn
import torch
torch.set_default_dtype(torch.float64)
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.fc3 = nn.Linear(16, output_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x.reshape(x.shape[0], -1)
        output = self.activation(self.fc1(input))
        output = self.activation(self.fc2(output))
        output = self.fc3(output)

        return output