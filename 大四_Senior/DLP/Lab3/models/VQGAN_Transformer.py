import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        
        z_map, z_indices, loss = self.vqgan.encode(x)
        z_indices = z_indices.view(z_map.shape[0], -1)
        return z_map, z_indices
        
        
        #raise Exception('TODO2 step1-1!')
        #return None
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            #raise Exception('TODO2 step1-2!')
            return lambda r : 1 - r
        elif mode == "cosine":
            #raise Exception('TODO2 step1-2!')
            return lambda r : np.cos(r * np.pi / 2)
        elif mode == "square":
            #raise Exception('TODO2 step1-2!')
            return lambda r : 1 - r**2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        
        # z_indices=ground truth
        # logits=transformer predict the probability of tokens
        
        _, z_indices = self.encode_to_z(x)
        mask_tk = torch.ones(z_indices.shape, device = z_indices.device).long()*self.mask_token_id
        mask = torch.bernoulli(0.5*torch.ones(z_indices.shape, device = z_indices.device)).long()
    
        new_ind = mask*mask_tk + (1-mask)*z_indices
        logits = self.transformer(new_ind)

        return logits, z_indices
    
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, mask_num, ratio, mask_func):
        

        masked_z_indices  = mask * self.mask_token_id + (~mask) * z_indices
        logits = self.transformer(masked_z_indices)
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        prob_dist = torch.softmax(logits, dim=-1)
        
        #FIND MAX probability for each token value
        while True:
            z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
            if torch.all(z_indices_predict != self.mask_token_id):
                break

        z_indices_predict = mask * z_indices_predict + (~mask) * z_indices
        z_indices_predict_prob = prob_dist.gather(-1, z_indices_predict.unsqueeze(-1)).squeeze(-1)
        z_indices_predict_prob = torch.where(mask, z_indices_predict_prob, torch.zeros_like(z_indices_predict_prob) + torch.inf)
    
        mask_r = self.gamma_func(mask_func)(ratio)
        mask_l = torch.floor(mask_r * mask_num).long()
        
        #predicted probabilities add temperature annealing gumbel noise as confidence
        eps = 1e-6
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob).clamp(eps, 1 - eps)))

        temperature = self.choice_temperature * (1 - mask_r)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        sort_conf = torch.sort(confidence, dim=-1)[0]
        
        cut_off = sort_conf[:, mask_l].unsqueeze(-1)
        mask_bc=(confidence < cut_off ) 
        
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
