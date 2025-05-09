import torch
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import math

import torch.nn.functional as F
from ddpm import Diffusion

class DiffusionImplicit(Diffusion):
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDIM', img_size=16, device="cuda"):
        """
        T : total diffusion steps (X_T is pure noise N(0,1))
        beta_start: value of beta for t=0
        b_end: value of beta for t=T
        diff_type: {DDIM, DDIM-cg, DDIM-cFg}    
            NOTE: 
                * DDIM: Traditional DDIM (https://arxiv.org/pdf/2006.11239.pdf)
                * DDIM-cg: DDIM with classifier guidance (https://arxiv.org/pdf/2105.05233.pdf)
                * DDIM-cFg: DDIM with classifier FREE guidance (https://arxiv.org/pdf/2207.12598.pdf)    
        """
        super().__init__(T=T, beta_start=beta_start, beta_end=beta_end, diff_type=diff_type, img_size=img_size, device=device)
    
    def p_sample(self, model, x_t, t, y=None, gradient_scale=1):
        """
        Sample from p(x{t-1} | x_t) using the reverse process and model
        """
        mean, std = self.p_mean_std(model, x_t, t, y, gradient_scale=gradient_scale)
        if t[0] > 1:
            noise = torch.randn_like(x_t, device=self.device)
        else:
            noise = torch.zeros_like(x_t, device=self.device)
        return mean + std * noise


    def p_sample_loop(self, model, batch_size, partly_noised, L, timesteps_to_save=None, y=None, verbose=True,gradient_scale=1):
        """
        y is class label
        """
        if verbose:
            logging.info(f"Sampling {batch_size} new images....")
            pbar = tqdm(reversed(range(1, L)), position=0, total=L-1)
        else :
            pbar = reversed(range(1, L))
            
        model.eval()
        if timesteps_to_save is not None:
            intermediates = []
        with torch.no_grad():
            # x = torch.randn((batch_size, 3, int(self.img_size[0]), int(self.img_size[1]))).to(self.device)
            x = partly_noised
            for i in pbar:
                t = (torch.ones(batch_size) * i).long().to(self.device)
                # T-1, T-2, .... 0
                x = self.p_sample(model, x, t, y, gradient_scale=gradient_scale)

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x
        
        
        
        
        
        
        
        
        
    # def p_sample(self, model, x_t, t, y=None, scale=1):
    #     """
    #     Sample from p(x{t-1} | x_t) using the reverse process and model
    #     """
    #     mean, _ = self.p_mean_std(model, x_t, t)
    #     std = 0

    #     # HINT: Having calculate the mean and std of p(x{x_t} | x_t), we sample noise from a normal distribution.
    #     # see line 3 of the Algorithm 2 (Sampling) at page 4 of the ddpm paper.
    #     noise = torch.randn_like(x_t)
        
    #     alpha = self.alphas[t][:, None, None, None] # match image dimensions
    #     alpha_bar = self.alphas_bar[t][:, None, None, None] # match image dimensions 
        
    #     x_t_prev = (1/torch.sqrt(alpha)) * \
    #                (x_t - ((1-alpha) / (torch.sqrt(1-alpha_bar)) * model(x_t, t))) + \
    #                 std*noise   
         
    #                 # Calculate x_{t-1}, see line 4 of the Algorithm 2 (Sampling) at page 4 of the ddpm paper.
    #     return x_t_prev


    # def p_sample_loop(self, model, batch_size, partly_noised, L, timesteps_to_save=None, y=None, verbose=True):
    #     """
    #     y is class label
    #     """
    #     if verbose:
    #         logging.info(f"Sampling {batch_size} new images....")
    #         pbar = tqdm(reversed(range(1, L)), position=0, total=L-1)
    #     else :
    #         pbar = reversed(range(1, L))
            
    #     model.eval()
    #     if timesteps_to_save is not None:
    #         intermediates = []
    #     with torch.no_grad():
    #         # x = torch.randn((batch_size, 3, int(self.img_size[0]), int(self.img_size[1]))).to(self.device)
    #         x = partly_noised
    #         for i in pbar:
    #             t = (torch.ones(batch_size) * i).long().to(self.device)
    #             # T-1, T-2, .... 0
    #             x = self.p_sample(model, x, t, y)
    #             if timesteps_to_save is not None and i in timesteps_to_save:
    #                 x_itermediate = (x.clamp(-1, 1) + 1) / 2
    #                 x_itermediate = (x_itermediate * 255).type(torch.uint8)
    #                 intermediates.append(x_itermediate)

    #     model.train()
    #     x = (x.clamp(-1, 1) + 1) / 2
    #     x = (x * 255).type(torch.uint8)

    #     if timesteps_to_save is not None:
    #         intermediates.append(x)
    #         return x, intermediates
    #     else :
    #         return x
