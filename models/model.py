import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.relu = nn.ReLU(inplace=True)

        self.linears = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.linears.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.linears.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.relu(layer(x))
        x = self.linears[-1](x)
        return x


if __name__ == "__main__":
    input_dim = 28*28
    output_dim = 10
    hidden_dims = [100, 50]
    model = FCN(input_dim, output_dim, hidden_dims)
    x = torch.rand(28*28)
    print(model(x))