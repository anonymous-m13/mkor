import torch

class AutoEncoder(torch.nn.Module):
    def __init__(self, img_size):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.img_size = img_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(img_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 9)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, img_size),
        )

    def forward(self, x):
        encoded = self.encoder(x.view(-1, self.img_size))
        decoded = self.decoder(encoded)
        return decoded


def autoencoder(img_size, **kwargs):
    img_size = img_size * img_size * 3
    return AutoEncoder(img_size)