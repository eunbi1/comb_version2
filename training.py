import os
from sampling import *
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10,CelebA, CIFAR100
import tqdm
import os
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import time
from model import *
from losses import *
from Diffusion import *
import numpy as np
import torch
from cifar10_model import *
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize





torch.multiprocessing.set_start_method('spawn')
from torchlevy import LevyStable
levy = LevyStable()

levy = LevyStable()

image_size = 28
channels = 1
batch_size = 128


def train(alpha=2, lr = 1e-4, batch_size=128, beta_min=0.1, beta_max = 20,
          n_epochs=15,num_steps=1000, datasets ='MNIST',path1 = None, path2= None, device='cuda',
          training_clamp=10, b= 0.8, c=0.2):
    sde = VPSDE(alpha, beta_min = beta_min, beta_max = beta_max, device=device, b=b,c=c)

    if device == 'cuda':
        num_workers =0
    else:
        num_workers = 4

    if datasets =="MNIST":
        dataset = MNIST('/home/eunbiyoon/comb', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,generator=torch.Generator(device=device))
        image_size = 28
        channels = 1
        score_model1 = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,))
        score_model2 = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,))

    if datasets == "CelebA":
        image_size = 32
        channels = 3

        transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),])
        dataset = CelebA('/scratch/private/eunbiyoon/comb', transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,generator=torch.Generator(device=device))

        score_model = Model(resolution=image_size, in_channels=channels,out_ch=channels)

    if datasets =='CIFAR10':
        dataset = CIFAR10('/home/eunbiyoon/comb', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,generator=torch.Generator(device=device))
        image_size = 32
        channels = 3
        score_model1 = Model()
        score_model2 = Model()


    score_model1 = score_model1.to(device)
    score_model2 = score_model2.to(device)
    if path1 and path2:
      ckpt1 = torch.load(path1, map_location=device)
      score_model1.load_state_dict(ckpt1,  strict=False)
      ckpt2 = torch.load(path2, map_location=device)
      score_model2.load_state_dict(ckpt2, strict=False)

    parameters = list(score_model1.parameters()) + list(score_model2.parameters())
    optimizer = Adam(parameters, lr=lr)


    L = []
    counter = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        counter += 1
        avg_loss = 0.
        num_items = 0
        i=0
        for x,y in data_loader:
            x = 2*x - 1
            x = x.to(device)
            n= x.size(0)
            e_B = torch.clamp(torch.randn(size=x.shape), -3,3)
            e_L = torch.clamp(levy.sample(alpha, 0, size=x.shape ).to(device),-training_clamp,training_clamp)

            t = torch.rand(n).to(device)*(1-1e-5)+1e-5
            # t = torch.randint(low=0, high=999, size=(n // 2 + 1,) ).to(device)
            # t = torch.cat([t, 999- t - 1], dim=0)[:n]
            # t = (t+1)/1000
            loss = loss_fn(score_model1, score_model2, sde, x, t, e_B,e_L, num_steps=num_steps)

            optimizer.zero_grad()
            loss.backward()
            print(f'{epoch} th epoch {i} th step loss: {loss}')
            i +=1

            optimizer.step()

            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        t1 = time.time()
        L.append(avg_loss / num_items)
        print(f'{epoch} th epoch loss: {avg_loss / num_items}')
        print('Running time:', t1-t0)
        dir_path = "new100"+ str(datasets) + str(f'clamp{training_clamp}') +str(f'b{b}c{c}')+ str(f'{alpha}_{beta_min}_{beta_max}')
        dir_path = os.path.join('/home/eunbiyoon/comb', dir_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        dir_path2=os.path.join(dir_path,'ckpt')
        if not os.path.isdir(dir_path2):
            os.mkdir(dir_path2)
        ckpt_name1 = '1'+ str(datasets)+str(f'clamp{training_clamp}')+ str(f'epoch{epoch}_')+str(f'{alpha}_{beta_min}_{beta_max}.pth')
        ckpt_name1=os.path.join(dir_path2, ckpt_name1)
        torch.save(score_model1.state_dict(),ckpt_name1)
        ckpt_name2 = '2'+ str(datasets) + str(f'clamp{training_clamp}') + str(f'epoch{epoch}_') + str(f'{alpha}_{beta_min}_{beta_max}.pth')
        ckpt_name2 = os.path.join(dir_path2, ckpt_name2)
        torch.save(score_model2.state_dict(), ckpt_name2)

        sample(dir_path=dir_path, alpha=sde.alpha, path1=ckpt_name1, path2=ckpt_name2,
                     beta_min=beta_min, beta_max=beta_max, sampler='pc_sampler2', batch_size=64, num_steps=1000, LM_steps=50,
                     Predictor=True, Corrector=False, trajectory=False, clamp=20, initial_clamp=20,
                     clamp_mode="constant",b=b,c=c,
                     datasets=datasets, name=epoch)

    #torch.save(score_model.state_dict(), name)
