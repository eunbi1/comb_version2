from torchvision.utils import make_grid
import tqdm
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
from losses import *
import numpy as np
import torch
from Diffusion import *
from torchlevy import LevyStable
levy = LevyStable()



## Sample visualization.

def visualization(samples, sample_batch_size=64):
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()


def gamma_func(x):
    return torch.tensor(gamma(x))


def get_discrete_time(t, N=1000):
    return N * t


def ddim_score_update2(score_model1, score_model2, sde, x_s, s, t, h=0.6, clamp = 10, device='cuda'):
    sigma1, sigma2 = sde.marginal_std(s)

    score_s1 = score_model1(x_s, s)*torch.pow(sigma2,-1)[:,None,None,None]
    score_s2 = score_model2(x_s, s) * torch.pow(sigma2, -(sde.alpha-1))[:, None, None, None]
    #*torch.pow(sigma,-1)[:,None,None,None]
    time_step = s-t
    beta_step = sde.beta(s)*time_step
    x_coeff = 1 + beta_step/2
    score_coeff1 = sde.b**2*beta_step
    score_coeff2 = sde.c**sde.alpha*sde.alpha**2/2*beta_step
    noise_coeff1 = sde.b*torch.pow(beta_step, 1/2)
    noise_coeff2 = sde.c*torch.pow(sde.alpha/2*beta_step, 1 / sde.alpha)

    e_B = torch.clamp(torch.randn(size =x_s.shape).to(device),-3,3)
    e_L = torch.clamp(levy.sample(sde.alpha, 0, size=x_s.shape ).to(device),-clamp,clamp)

    x_t = x_coeff[:, None, None, None] * x_s + score_coeff1[:, None, None, None]* score_s1+score_coeff2[:,None,None,None]* score_s2 + noise_coeff1[:,None,None,None]*e_B + noise_coeff2[:,None,None,None]*e_L
    #print('score_coee', torch.min(score_coeff), torch.max(score_coeff))
    #print('noise_coeff',torch.min(noise_coeff), torch.max(noise_coeff))
    #print('x_coeff', torch.min(x_coeff), torch.max(x_coeff))
    
    print('x_s range', torch.min(x_s), torch.max(x_s))
    print('x_t range', torch.min(x_t), torch.max(x_t))
    print('x_s mean', torch.mean(x_s))
    print('x_t mean', torch.mean(x_t))


    #print('x coeff adding', torch.min(x_coeff[:, None, None, None] * x_s), torch.max(x_coeff[:, None, None, None] * x_s))
    #print('score adding',torch.min(score_coeff[:, None, None, None] * score_s), torch.max(score_coeff[:, None, None, None] * score_s) )
    #print('noise adding', torch.min(noise_coeff[:, None, None,None] * e_L), torch.max(noise_coeff[:, None, None,None] * e_L))

    return x_t


def pc_sampler2(score_model1, score_model2,
                sde,
                alpha,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                x_0= None,
                Predictor=True,
                Corrector=False, trajectory=False,
                clamp = 10,
                initial_clamp =3, final_clamp = 1,
                datasets="MNIST", clamp_mode = 'constant'):
    t = torch.ones(batch_size, device=device)
    if datasets =="MNIST":
        sigma1 , sigma2 = sde.marginal_std(t)
        e_B= torch.randn((batch_size, 1, 28,28)).to(device)
        e_L = levy.sample(alpha, 0, (batch_size, 1, 28,28)).to(device)
        x_s = sigma2[:,None,None,None]*torch.clamp(e_L,-initial_clamp,initial_clamp)+sigma1[:,None,None,None]*e_B
    elif datasets =="CIFAR10":
        sigma1, sigma2 = sde.marginal_std(t)
        e_B = torch.clamp(torch.randn((batch_size, 3, 32, 32)).to(device),-3,3)
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32,32)).to(device)
        x_s = sigma2[:,None,None,None]*torch.clamp(e_L,-initial_clamp,initial_clamp)+sigma1[:,None,None,None]*e_B

    elif datasets =="CelebA":
        e_B = torch.randn((batch_size, 3, 64, 64)).to(device)
        e_L = levy.sample(alpha, 0, (batch_size, 3, 64,34)).to(device)
        x_s = sigma2[:,None,None,None]*torch.clamp(e_L,-initial_clamp,initial_clamp)+sigma1[:,None,None,None]*e_B
    if x_0 :
        x_s = x_0
        
    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.linspace(1., 1e-5, num_steps)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0])*t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0]*(clamp-final_clamp)+final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0],1/2)*(clamp-final_clamp)+final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0]**2*(clamp-final_clamp)+final_clamp

            if Predictor:
                 x_s = ddim_score_update2(score_model1,score_model2, sde, x_s, batch_time_step_s, batch_time_step_t, clamp = linear_clamp)
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t



    if trajectory:
        return samples
    else:
        return x_s


def ode_score_update(score_model1, score_model2, sde, x_s, s, t, clamp=3, h=0.4, return_noise=False):
    sigma1, sigma2 = sde.marginal_std(s)
    score_s1 = score_model1(x_s, s) * torch.pow(sigma2, -1)[:, None, None, None]
    score_s2 = score_model2(x_s, s) * torch.pow(sigma2, -(sde.alpha - 1))[:, None, None, None]
    time_step = s - t
    beta_step = sde.beta(s) * time_step

    #x_coeff = 1 + beta_step / alpha
    x_coeff =sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s), -1)
    lambda_s_1, lambda_s_2 = sde.marginal_lambda(s)
    lambda_t_1, lambda_t_2 = sde.marginal_lambda(t)

    h_t_1 = lambda_t_1 - lambda_s_1
    h_t_2 = lambda_t_2 - lambda_s_2


    time_step = s - t


    beta_step = sde.beta(s) * time_step
    x_coeff = sde.diffusion_coeff(t)*torch.pow(sde.diffusion_coeff(s), -1)

    score_coeff1 = -1/2*2* torch.pow(sde.marginal_std(s)[0], 1) * sde.marginal_std(t)[0] * (1 - torch.exp(h_t_1))
    score_coeff2 = -sde.alpha * torch.pow(sde.marginal_std(s)[1], sde.alpha - 1) * sde.marginal_std(t)[1] * (1 - torch.exp(h_t_2))

    x_t = x_coeff[:, None, None, None] * x_s + score_coeff1[:, None, None, None] * score_s1+ score_coeff2[:, None, None, None] * score_s2
    #score_t = score_model(x_t, t)*torch.pow(sde.marginal_std(t)[0],-2)[:,None,None,None]
    #second =gamma_func(sde.alpha-1)/gamma_func(sde.alpha/2)**2/h**(sde.alpha-2)*sde.alpha*((torch.exp(-lambda_t_2)*sde.marginal_std(t)[1]**(sde.alpha-1))[:,None,None,None]*score_t+(torch.exp(-lambda_s_2)*sde.marginal_std(s)[1]**(sde.alpha-1))[:,None,None,None]*score_s)/2*(t-s)[:,None,None,None]
    #x_t = x_coeff[:, None, None, None] * x_s + score_coeff1[:, None, None, None] * score_s + second
  
    print('x_s range', torch.min(x_s), torch.max(x_s))
    print('x_t range', torch.min(x_t), torch.max(x_t))

    return x_t

def ode_sampler(score_model1, score_model2,
                sde,
                alpha,
                batch_size,
                num_steps,
                device='cuda',
                eps=1e-4,
                x_0=None,
                Predictor=True,
                Corrector=False, trajectory=False,
                clamp=10,
                initial_clamp=3, final_clamp=1,
                datasets="MNIST", clamp_mode='constant'):
    t = torch.ones(batch_size, device=device)
    sigma1, sigma2 = sde.marginal_std(t)
    if datasets == "MNIST":
        sigma1 , sigma2 = sde.marginal_std(t)
        e_B= torch.randn((batch_size, 1, 28,28)).to(device)
        e_L = levy.sample(alpha, 0, (batch_size, 1, 28,28)).to(device)
        x_s = sigma2[:,None,None,None]*torch.clamp(e_L,-initial_clamp,initial_clamp)+sigma1[:,None,None,None]*e_B
    elif datasets == "CIFAR10":
        e_B = torch.clamp(torch.randn((batch_size, 3, 32, 32)).to(device),-3,3)
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
        x_s = sigma2[:,None,None,None]*torch.clamp(e_L,-initial_clamp,initial_clamp)+sigma1[:,None,None,None]*e_B

    elif datasets == "CelebA":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
    if x_0:
        x_s = x_0

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp


            x_s =ode_score_update(score_model1, score_model2, sde, x_s, batch_time_step_s, batch_time_step_t,
                                         clamp=linear_clamp)
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t

    if trajectory:
        return samples
    else:
        return x_s