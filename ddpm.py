import torch
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import math
import numpy as np 

import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cg', img_size=16,device="cuda"):
        """
        T : total diffusion steps (X_T is pure noise N(0,1))
        beta_start: value of beta for t=0
        b_end: value of beta for t=T
        diff_type: {DDPM, DPPM-cg, DDPM-cFg}    
            NOTE: 
                * DDPM: Traditional DDPM (https://arxiv.org/pdf/2006.11239.pdf)
                * DDPM-cg: DDPM with classifier guidance (https://arxiv.org/pdf/2105.05233.pdf)
                * DDPM-cFg: DDPM with classifier FREE guidance (https://arxiv.org/pdf/2207.12598.pdf)
        
        """

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.betas = self.get_betas().to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) # cumulative products of alpha
        self.diff_type = diff_type
        self.classifier = None 
        assert diff_type in {'DDPM', 'DDPM-cg', 'DDPM-cFg'}, 'Invalid diffusion type'
        print(f'Diffusion type: {diff_type}')


    def get_betas(self, schedule='linear'):
        if schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.T)
        else :
            raise NotImplementedError('Not implemented!')
    

    def q_sample(self, x, t):
        """
        x: input image (x0)
        t: timestep: should be torch.tensor

        Forward diffusion process
        q(x_t | x_0) = sqrt(alpha_hat_t) * x0 + sqrt(1-alpha_hat_t) * N(0,1)

        Returns q(x_t | x_0), noise
        """
        # print(x.shape) # 100, 3, 16, 16
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t])
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None] # match image dimensions
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar[t])
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]# match image dimensions

        noise = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise
    

    def classifier_conditioning(self, x_t, t, y):

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            logits = self.classifier(x_in, t) 
            log_probs = F.log_softmax(logits, dim=-1) # B x 5
            y_log_probs = log_probs[range(len(logits)), y.view(-1)] # get the logits that correspont to the input class y
            gradient = torch.autograd.grad(y_log_probs.sum(), x_in)[0]
            
            # new_mu = mean + std * gradient.float()
            # new_mu = mean - std * torch.sqrt(1 - self.alphas_bar[t]) * gradient.float()
            
            # "Diffusion models beat GANs for image generation" - Algorithm 1

        # return new_mu
        return gradient


    def p_mean_std(self, model, x_t, t, y=None, gradient_scale=1):
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        Implements Algrorithm 1 from the DDPM-cg paper at page 4
        """
        alpha = self.alphas[t][:, None, None, None] # match image dimensions
        alpha_bar = self.alphas_bar[t][:, None, None, None] # match image dimensions 
        beta = self.betas[t][:, None, None, None] # match image dimensions
        # Functionality for classifier guidance
        if y is not None and self.diff_type=='DDPM-cg':
            gradient = self.classifier_conditioning(x_t, t, y)    
            predicted_noise = model(x_t, t)   
            # print("predicted noise", predicted_noise)
            predicted_noise -= gradient_scale * torch.sqrt(1 - alpha_bar) * gradient
            # print("predicted noise", predicted_noise)
            
        # mean = 1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) 
        # std = torch.sqrt(beta)
        mean = torch.sqrt(self.alphas_bar[t-1]) * (x_t - torch.sqrt(1-alpha_bar) * predicted_noise)/torch.sqrt(alpha_bar) + \
               torch.sqrt(1 - self.alphas_bar[t-1])*predicted_noise 
        std  = 0
         
        return mean, std


    def p_sample(self, model, x_t, t, y=None, gradient_scale=1):
        """
        Sample from p(x{t-1} | x_t) using the reverse process and model
        """
        mean, std = self.p_mean_std(model, x_t, t, y, gradient_scale)
        if t[0] > 1:
            noise = torch.randn_like(x_t, device=self.device)
        else:
            noise = torch.zeros_like(x_t, device=self.device)
        return mean + std * noise


    def p_sample_loop(self, model, batch_size, timesteps_to_save=None, y=None, verbose=True,gradient_scale=1):
        """
        y is class label
        """
        if verbose:
            logging.info(f"Sampling {batch_size} new images....")
            pbar = tqdm(reversed(range(1, self.T)), position=0, total=self.T-1)
        else :
            pbar = reversed(range(1, self.T))
            
        model.eval()
        if timesteps_to_save is not None:
            intermediates = []
        with torch.no_grad():
            x = torch.randn((batch_size, 3, int(self.img_size[0]), int(self.img_size[1]))).to(self.device)
            for i in pbar:
                t = (torch.ones(batch_size) * i).long().to(self.device)
                # T-1, T-2, .... 0
                x = self.p_sample(model, x, t, y, gradient_scale)
                if timesteps_to_save is not None and i in timesteps_to_save:
                    x_itermediate = (x.clamp(-1, 1) + 1) / 2
                    x_itermediate = (x_itermediate * 255).type(torch.uint8)
                    intermediates.append(x_itermediate)

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        if timesteps_to_save is not None:
            intermediates.append(x)
            return x, intermediates
        else :
            return x
    

    def sample_timesteps(self, batch_size, upper_limit=None):
        """
        Sample timesteps uniformly for training
        """
        if upper_limit is None:
            return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)
        else :
            return torch.randint(low=1, high=upper_limit, size=(batch_size,), device=self.device)
