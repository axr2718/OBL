import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10, bias=False)
        self.fc2 = nn.Linear(10, output_dim, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x.reshape(x.size(0), -1)
        output = self.activation(self.fc1(input))
        output = self.fc2(output)

        return output
        