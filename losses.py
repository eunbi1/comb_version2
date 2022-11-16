import torch
import copy
import time
from scipy.special import gamma
#from torchlevy import levy, LevyGaussianz
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import torchlevy
import torch.nn.functional as F

def gamma_fn(x):
    return torch.tensor(gamma(x))

import torch
from torchlevy import levy_gaussian_score


def loss_fn(model1, model2, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            e_B,
            e_L: torch,
            num_steps=1000, type="cft", mode='only'):
    sigma1, sigma2 = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)

    #levy1 = LevyGaussian(alpha=2,  sigma_1=sigma_1, sigma_2=sigma_2)
    #levy2 = LevyGaussian(alpha=sde.alpha,  sigma_1=sigma_1, sigma_2=sigma_2)
    # score1 = levy1.score(noise, 2, sigma1=sigma_1, sigma2=sigma_2)
    # score2 = levy2.score(noise, alpha=sde.alpha, sigma1=sigma_1, sigma2=sigma_2)
    #score = get_approx_score(noise, alpha, sigma_1,sigma_2)

    noise = (e_B * sigma1[:, None, None, None] + e_L * sigma2[:, None, None, None])*torch.pow(sigma2, -1)[:,None,None,None]
    # print('noise', torch.max(noise), torch.min(noise))
    S1=[]
    # for i, sB, sL in zip(range(len(x0)), sigma1,sigma2):
    #     n = noise[i]
    #     levy1 = LevyGaussian(alpha=2, sigma_1=sB, sigma_2=sL)
    #     score1 = levy1.score(n)
    #     S1.append(score1)
    score1 = levy_gaussian_score(sde.alpha, noise, sigma1, sigma2, mode='score')
    score2 = levy_gaussian_score(sde.alpha, noise, sigma1, sigma2)
    x_t = x_coeff[:, None, None, None] * x0 + e_B*sigma1[:,None,None,None]+ e_L * sigma2[:, None, None, None]
    output1 = model1(x_t, t)
    output2 = model2(x_t,t)
    weight1 = (output1 - score1)
    weight2 = (output2-score2)
    loss1= (weight1).square().sum(dim=(1, 2, 3)).mean(dim=0)
    loss2 = (weight2).square().sum(dim=(1, 2, 3)).mean(dim=0)

    loss = loss1 + loss2
    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    #print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #print('weight', torch.min(weight), torch.max(weight))

    return  loss



