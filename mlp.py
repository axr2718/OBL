import torch.nn as nn
import torch
torch.set_default_dtype(torch.float64)
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units, bias=False)
        self.fc2 = nn.Linear(hidden_units, output_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x.reshape(x.shape[0], -1)
        output = self.activation(self.fc1(input))
        output = self.fc2(output)

        return output