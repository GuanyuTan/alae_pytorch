import os
import sys
from datetime import datetime
import torch
import parser
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

arser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self)-> None:
        super(Generator, self).__init__()

        def block(n_in, n_out, normalize=True):
            layers = [nn.Linear(n_in, n_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(n_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(100, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0),*img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        def block(n_in, n_out, normalize=True):
            layers = [nn.Linear(n_in, n_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(n_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(
            *block(int(np.prod(img_shape), 512)),
            *block(int(512, 256)),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        # whether the input image is generated or not
        out = self.model(img)
        return out
    
now = datetime.now()

save_dir = os.path.join('outputs', now.strftime('%y-%m-%d'), now.strftime('%h-%m-%s'))
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

dataloader = DataLoader(
    datasets.MNIST(
        train=True,
        download=True,
        transform=[
            transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
        ]
    ),
    batch_size = opt.batch_size,
    shuffle=True,
)


adversarial_loss = torch.nn.BCELoss()
generator = Generator()
discriminator = Discriminator()

cuda = torch.cuda.is_available()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        real = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # update for generator first
        optimizer_G.zero_grad()
        #  generate noise for our generator input
        z = Variable(Tensor(np.random.normal(0,1,[imgs.shape[0], opt.latent_dim])))

        gen_imgs = generator(z)

        generator_loss = adversarial_loss(discriminator(gen_imgs), real)
        generator_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
        discriminator_loss = (real_loss+fake_loss)/2
        discriminator_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{opt.n_epoch}] [Batch {i}/{len(dataloader)}] [Discriminator Loss: {discriminator_loss.item()}] [Generator Loss: {generator_loss.item()}]")

torch.save(generator.parameters(), os.path.join(save_dir, 'generator.pt'))
torch.save(generator.parameters(), os.path.join(save_dir, 'discriminator.pt'))