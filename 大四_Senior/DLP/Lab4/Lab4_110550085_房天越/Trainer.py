import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

from torch.utils import tensorboard

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.iter = current_epoch + 1
        self.kl_anneal_type = args.kl_anneal_type
        
        if self.kl_anneal_type == 'Cyclical':
            self.L = self.frange_cycle_linear(args.num_epoch, start=0.0, stop=1.0, n_cycle=args.kl_anneal_cycle, ratio=args.kl_anneal_ratio)
        elif self.kl_anneal_type == 'Monotonic':
            self.L = self.frange_cycle_linear(args.num_epoch, start=0.0, stop=1.0, n_cycle=1, ratio=args.kl_anneal_ratio)
        elif self.kl_anneal_type == 'None':
            self.L = np.ones(args.num_epoch+1)
        else:
            self.L = np.ones(args.num_epoch+1)
            
        
        # raise NotImplementedError
        
    def update(self):
        # TODO
        self.iter += 1
        # raise NotImplementedError
    
    def get_beta(self):
        # TODO
        return self.L[self.iter]
        # raise NotImplementedError

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        # Adapted from the Repo of the Paper of Cyclical Annealing
        # https://github.com/haofuml/cyclical_annealing
        L = np.ones(n_iter+1)*stop
        period = n_iter // n_cycle
        step = (stop - start) / (period * ratio)
        for cycle in range(n_cycle):
            v = start
            i = 0
            while v <= stop and int(i+(cycle*period)) < n_iter:
                L[i+(cycle*period)] = v
                v += step
                i += 1
                
        return L
        
        # raise NotImplementedError
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.writer = tensorboard.SummaryWriter(f"logs/kl-type_{args.kl_anneal_type}_tfr_{args.tfr}_teacher-decay_{args.tfr_d_step}")
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            train_loss = []
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                train_loss.append(loss.detach().cpu())
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            self.writer.add_scalar('loss/train', np.mean(train_loss), self.current_epoch)
            self.writer.add_scalar('tfr', self.tfr, self.current_epoch)
            self.writer.add_scalar('beta', beta, self.current_epoch)
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        val_loss = []
        psnr = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr_return = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            val_loss.append(loss.detach().cpu())
            psnr.append(psnr_return)
            
        val_loss = np.mean(val_loss)
        psnr = np.mean(psnr)
        
        self.writer.add_scalar('loss/validation',val_loss , self.current_epoch)
        self.writer.add_scalar('PSNR/validation',psnr , self.current_epoch)
            
            
            
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        batch_size = img.shape[0]
        losses = 0
        beta = self.kl_annealing.get_beta()
        # print("beta: ", beta)
        
        for i in range(batch_size):
            im = img[i]
            la = label[i]
            x_h = im[0].unsqueeze(0)
            
            mse = 0
            kl_div = 0
            
            for j in range(1, im.size(0)):
                if adapt_TeacherForcing:
                    last_im = im[j-1].unsqueeze(0)
                else:
                    last_im = x_h
            
                im_encoder = self.frame_transformation(im[j].unsqueeze(0))
                label_encoder = self.label_transformation(la[j].unsqueeze(0))
                
                z, mu, logvar = self.Gaussian_Predictor(im_encoder, label_encoder)
                
                last_im_encoder = self.frame_transformation(last_im).detach()
                decoder_out = self.Decoder_Fusion(last_im_encoder, label_encoder, z)
                x_h = self.Generator(decoder_out)
                
                mse += self.mse_criterion(x_h, im[j].unsqueeze(0))
                kl_div += kl_criterion(mu, logvar, batch_size)
                
            loss = mse + beta * kl_div
            self.optim.zero_grad()
            loss.backward()
            self.optimizer_step()
            losses += loss
            
        losses /= batch_size
        return losses
            
                
        
        
        
        # raise NotImplementedError
    
    def val_one_step(self, img, label):
        # TODO
        batch_size = img.shape[0]
        losses = 0
        beta = self.kl_annealing.get_beta()
        psnr = []
        
        for i in range(batch_size):
            im = img[i]
            la = label[i]
            x_h = im[0].unsqueeze(0)
            
            mse = 0
            kl_div = 0
            
            for j in range(1, im.size(0)):
                last_im = x_h.detach()
            
                im_encoder = self.frame_transformation(im[j].unsqueeze(0))
                label_encoder = self.label_transformation(la[j].unsqueeze(0))
                
                z, mu, logvar = self.Gaussian_Predictor(im_encoder, label_encoder)
                
                last_im_encoder = self.frame_transformation(last_im).detach()
                decoder_out = self.Decoder_Fusion(last_im_encoder, label_encoder, z)
                x_h = self.Generator(decoder_out)
                
                mse += self.mse_criterion(x_h, im[j].unsqueeze(0))
                kl_div += kl_criterion(mu, logvar, batch_size)
                psnr.append(Generate_PSNR(x_h, im[j].unsqueeze(0)).item())
                
            loss = mse + beta * kl_div
            losses += loss
            
        losses /= batch_size
        return losses, np.mean(psnr)
        
        # raise NotImplementedError
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        
        if self.current_epoch >= self.tfr_sde:
            if self.current_epoch % self.tfr_sde == 0:
                self.tfr -= self.tfr_d_step
                if self.tfr < 0:
                    self.tfr = 0
            
        #raise NotImplementedError
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()

import random, numpy as np, torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    
    main(args)
