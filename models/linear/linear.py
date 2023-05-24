import torch


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        self.layers.append(torch.nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.model(x.reshape(-1, self.input_dim))


def linear(img_size, depth=1, hidden_dim=1024, **kwargs):
    img_size = img_size * img_size * 3
    return LinearModel(img_size, img_size, depth, hidden_dim)
