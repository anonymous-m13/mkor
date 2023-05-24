import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

# Data Preprocessing

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Defining Model

class Autoencoder(nn.Module):

    def __init__(self, img_size, depth=1, size_change_rate=2, encoder_init_size=1000, encoder_final_size=30, transfer_function=nn.ReLU, **kwargs):
        super(Autoencoder,self).__init__()
        encoder_sizes = [max(int(encoder_init_size / size_change_rate ** i), encoder_final_size) for i in range(depth)] + [encoder_final_size]
        decoder_sizes = [min(int(encoder_final_size * size_change_rate ** i), img_size) for i in range(depth)] + [img_size]
        encoder_layers = []
        encoder_layers.append(nn.Linear(img_size, encoder_sizes[0]))
        encoder_layers.append(transfer_function())
        for i in range(1, len(encoder_sizes)):
            encoder_layers.append(nn.Linear(encoder_sizes[i-1], encoder_sizes[i]))
            if i < len(encoder_sizes) - 1:
                encoder_layers.append(transfer_function())
                
        decoder_layers = []
        for i in range(1, len(decoder_sizes)):
            decoder_layers.append(nn.Linear(decoder_sizes[i-1], decoder_sizes[i]))
            if i < len(decoder_sizes) - 1:
                decoder_layers.append(transfer_function())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        

    def forward(self,x):
        input_shape = x.shape
        x = x.view(input_shape[0], -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(input_shape)
        return x

def autoencoder(**kwargs):
    model = Autoencoder(**kwargs)
    return model


if __name__ == "__main__":
    model_depth_multiplier = 1
    encoder_sizes = [1000] +  [500] * model_depth_multiplier + [250, 30]
    decoder_sizes = [250] +  [500] * model_depth_multiplier + [1000]

    # Defining Parameters

    num_epochs = 5
    batch_size = 128
    model = Autoencoder(encoder_sizes, decoder_sizes).cpu()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = Variable(img).reshape(img.shape[0], -1).cpu()
            # ===================forward=====================
            output = model(img)
            loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))