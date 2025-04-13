import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20, bias=False)
        self.fc2 = nn.Linear(20, output_dim, bias=False)
        #self.fc3 = nn.Linear(10, output_dim, bias=False)
        #self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x.reshape(x.shape[0], -1)
        output = self.fc1(input)
        #output = self.activation(self.fc1(input))
        #output = self.activation(self.fc2(output))
        #output = self.activation(self.fc2(output))
        output = self.fc2(output)

        return output
        
        #93.0300 | Precision 92.9258 | Recall 92.9186 | F1: 92.9197