import torch


class Identity(torch.nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.img_size = img_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(img_size, img_size, bias=False),
        )

    def forward(self, x):
        encoded = self.encoder(x.view(-1, self.img_size))
        return encoded


def identity(img_size, **kwargs):
    img_size = img_size * img_size * 3
    return Identity(img_size)