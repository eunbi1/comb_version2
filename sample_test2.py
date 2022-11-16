import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from sampling import *
from training import *
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import random
import torch.backends.cudnn as cudnn
import torch
import numpy as np
a=1
torch.manual_seed(a)
torch.cuda.manual_seed(a)
torch.cuda.manual_seed_all(a)
np.random.seed(a)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(a)



if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
"""
for epoch in torch.arange(5,51,5):
    score_model = train(alpha=1.9, datasets= "CelebA", training_clamp=3,
          beta_min=0.1, beta_max=20, n_epochs=epoch)
    samples = sample(alpha=1.9, score_model=score_model, name=str(f'epoch{epoch}'),
                             beta_min=0.1, beta_max=7.5, sampler='pc_sampler2', batch_size=64, num_steps=1000,
                             LM_steps=10,
                             Predictor=True, Corrector=False, trajectory=False, clamp=3, initial_clamp=3,
                             clamp_mode="constant", datasets="CIFAR10")

"""
for epoch in torch.arange(4,5):
  path =f"/scratch/private/eunbiyoon/Levy_motion-/CelebAclamp3.0epoch{epoch}_1.9_0.1_7.5.pth"
  samples = sample(dir_path='/home/eunbiyoon/comb',alpha=1.8,
                   path1='/home/eunbiyoon/comb/new100CIFAR10clamp20b0.5c0.51.8_0.1_20/ckpt/1CIFAR10clamp20epoch4_1.8_0.1_20.pth',
                   path2 = '/home/eunbiyoon/comb/new100CIFAR10clamp20b0.5c0.51.8_0.1_20/ckpt/2CIFAR10clamp20epoch4_1.8_0.1_20.pth',
                   beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=64, num_steps=1000, LM_steps=50,
                   Predictor=True, Corrector=False, trajectory=False, clamp=20, initial_clamp=20, clamp_mode="constant",
                   datasets="CIFAR10", name=epoch.item(),b=0.5,c=0.5)

# for epoch in torch.arange(4,5):
#   path =f"/scratch/private/eunbiyoon/Levy_motion-/CelebAclamp3.0epoch{epoch}_1.9_0.1_7.5.pth"
#   samples = sample(alpha=1.8, path='/home/eunbiyoon/Levy_combined/CIFAR10clamp101.8_0.121_20.pth',
#                    beta_min=0.121, beta_max=20, sampler='pc_sampler2', batch_size=64, num_steps=1000, LM_steps=50,
#                    Predictor=True, Corrector=False, trajectory=True, clamp=10, initial_clamp=10, clamp_mode="constant",
#                    datasets="CIFAR10", name=epoch.item())

# for epoch in torch.arange(4,5):
#   path =f"/scratch/private/eunbiyoon/Levy_motion-/CelebAclamp3.0epoch{epoch}_1.9_0.1_7.5.pth"
#   samples = sample(alpha=1.8, path='/home/eunbiyoon/Levy_combined/MNISTclamp101.8_0.1_20.pth',
#                    beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=64, num_steps=1000, LM_steps=50,
#                    Predictor=True, Corrector=False, trajectory=False, clamp=10, initial_clamp=10, clamp_mode="constant",
#                    datasets="MNIST", name=epoch.item())

"""
print(device)
for initial_clamp in torch.arange(1, 3, 0.2):
    for clamp in torch.arange(1,3,0.2):
        for clamp_mode in ['root']:
            print(f'initial clamp{initial_clamp}, clamp {clamp}, model {clamp_mode}')
            samples = sample(alpha=1.9, path='/scratch/private/eunbiyoon/Levy_motion-/CelebAclamp3.01.9_0.1_7.5.pth',
                             beta_min=0.1, beta_max=7.5, sampler='pc_sampler2', batch_size=64, num_steps=1000,
                             LM_steps=10,
                             Predictor=True, Corrector=False, trajectory=False, clamp=clamp, initial_clamp=initial_clamp,
                             clamp_mode=clamp_mode, datasets="CelebA")

"""
"""
for initial_clamp in torch.arange(1, 3, 0.2):
    for clamp in torch.arange(1,3,0.2):
        for clamp_mode in ['constant', 'linear']:
            print(f'initial clamp{initial_clamp}, clamp {clamp}, model {clamp_mode}')
            samples = sample(alpha=1.9, path='/scratch/private/eunbiyoon/Levy_motion-/CelebAclamp3.01.9_0.1_7.5.pth',
                             beta_min=0.1, beta_max=7.5, sampler='pc_sampler2', batch_size=64, num_steps=1000,
                             LM_steps=10,
                             Predictor=True, Corrector=False, trajectory=False, clamp=clamp, initial_clamp=initial_clamp,
                             clamp_mode=clamp_mode, datasets="CelebA")
"""
#train(beta_min=0.1, alpha=1.9,beta_max=20,n_epochs=1000)
#train(beta_min=1, alpha=1.9, abeta_max=20,n_epochs=1000)
"""
samples = sample(alpha=1.9, path ='/scratch/private/eunbiyoon/Levy_motion-/CelebAclamp3.01.9_0.1_7.5.pth',
            beta_min=0.1, beta_max=7.5, sampler='pc_sampler2', batch_size=64, num_steps=1000,LM_steps=10,
                Predictor=True, Corrector=False, trajectory = False, clamp =1.5, initial_clamp =1.5, clamp_mode = "constant", datasets="CelebA")
#sample(alpha=1.9, path ="/home/eunbiyoon/Levy_motion-/MNIST1.9_0.1_20.pth",beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=64,num_steps=1000,LM_steps=20,Predictor=True, Corrector=False)
"""
