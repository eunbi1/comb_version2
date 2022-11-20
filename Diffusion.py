import torch
import math
from torch.autograd import Variable
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=20, schedule='cosine', b=0.8,c=0.2, device=device):
            self.beta_0 = beta_min
            self.beta_1 = beta_max
            self.cosine_s = 0.008
            self.schedule = schedule
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.alpha = alpha
            self.b =b

            self.c = c
    def beta(self, t):
            if self.schedule == 'linear':
                beta = (self.beta_1 - self.beta_0) * t + self.beta_0
            elif self.schedule == 'cosine':
                beta = math.pi / 2 *2 / (self.cosine_s + 1) * torch.tan(
                    (t + self.cosine_s) / (1 + self.cosine_s) * math.pi / 2)
            beta = torch.clamp(beta, 0, 20)
            return beta

    def marginal_log_mean_coeff(self, t):
        if self.schedule =='linear':
          log_alpha_t = - 1 / (2 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0

        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        sigma1 = self.b*torch.pow(1. - torch.exp(2 * self.marginal_log_mean_coeff(t)), 1/2)
        sigma2 = self.c*torch.pow(1. - torch.exp(self.alpha * self.marginal_log_mean_coeff(t)), 1 / self.alpha)
        return sigma1, sigma2

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma1, log_sigma2 = torch.log(self.b*torch.pow(1. - torch.exp(2 * log_mean_coeff), 1 / 2)), torch.log(self.c*torch.pow(1. - torch.exp(self.alpha * log_mean_coeff), 1 / self.alpha))
        return log_mean_coeff - log_sigma1, log_mean_coeff-log_sigma2

