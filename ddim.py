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
        
    
