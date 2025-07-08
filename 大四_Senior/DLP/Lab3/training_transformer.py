import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        
        self.writer = SummaryWriter("log/")
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
            self.model.transformer.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint_path}")

        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch, args):
        #pass
        train_bar = tqdm(enumerate(train_loader))
        losses = []
        
        self.model.train()
        for i, data in train_bar:
            data = data.to(args.device)
            logits, z_indices = self.model(data)
            
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), z_indices.view(-1))
            loss.backward()
            losses.append(loss.item())
            if i % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            train_bar.set_description_str(f"Epoch {epoch} | Iteration {i+1} / {len(train_loader)} | Loss: {np.mean(losses):.4f}")   
            
        train_loss = np.mean(losses)
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        return train_loss    

    def eval_one_epoch(self, val_loader, epoch, args):
        valid_bar = tqdm(enumerate(val_loader))
        losses = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data in valid_bar:
                data = data.to(args.device)
                logits, z_indices = self.model(data)
                
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), z_indices.view(-1))
                losses.append(loss.item())
                valid_bar.set_description_str(f"Validation | Iteration {i} / {len(val_loader)} | Loss: {np.mean(losses):.4f}")
        
        valid_loss = np.mean(losses)
        self.writer.add_scalar("Loss/valid", valid_loss, epoch)
        return valid_loss
        
        #pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformers_checkpoints/best_train.pth', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=0.5, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(default: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5: 
    beste_train = np.inf
    beste_val = np.inf

    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args)
        #torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/train_{epoch}.pt")
        if train_loss < beste_train:
            beste_train = train_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/best_train.pth")
            print(f"best train loss: {beste_train:.4f}")
        if val_loss < beste_val:
            beste_val = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/best_val.pth")
            print(f"best val loss: {beste_val:.4f}")
        #pass
    
    