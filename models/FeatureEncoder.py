import torch
import torch.nn as nn
from torch import optim
from einops import rearrange
from models.layers import casual_cnn, triple_loss
import numpy as np
import os
import time
class FeatureEncoder(nn.Module):
    def __init__(self, checkpoints = './checkpoints', in_channels=38, channels=40, negative_penalty=1,
                 epochs=5, lr=0.001, nb_random_samples=5, 
                 depth=10, 
                 reduced_size=384, out_channels=768, kernel_size=3, cuda=False, gpu=0):
        super(FeatureEncoder, self).__init__()
        self.cuda = cuda
        self.gpu = gpu
        self.epochs = epochs
        self.checkpoints = checkpoints
        # channels = in_channels
        # reduced_size = in_channels
        # out_channels = in_channels
        self.loss = triple_loss.TripleLoss(nb_random_samples, negative_penalty)
        self.encoder = casual_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels, kernel_size
        )
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.encoder.double()


    def forward(self, x, patch_size=25):
        # x: (batch_size x length x nvars)
        # x = rearrange(x, 'b (n s) m -> b n s m', s=patch_size)
        # means = x.mean(2, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(
        #     torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
        # x /= stdev
        # x = rearrange(x, 'b n s m -> b (n s) m')  # (batch_size, length, d_model)


        x = x.permute(0, 2, 1)
        x = self.encoder(x) # (batch_size x d_model x length)
        x = x.permute(0, 2, 1) # (batch_size x length x d_model)

        # x = rearrange(x, 'b (n s) m -> b n s m', s=patch_size)
        # # print(dec_out.shape)
        # x = x * \
        #           (stdev[:, :, 0, :].unsqueeze(2).repeat(
        #               1, 1, patch_size, 1))
        # x = x + \
        #           (means[:, :, 0, :].unsqueeze(2).repeat(
        #               1, 1, patch_size, 1))
        # x = rearrange(x, 'b n s m -> b (n s) m')

        return x
        

    def fit(self, train_loader, setting, patch_size=25):
        # X: (train_size x nvars x num_patch x patch_length)
        print("Start train feature encoder...")

        i = 0  # Number of performed optimization steps

        # Encoder training
        best_loss = float('inf')
        early_stopper = 0
        
        train_total_t = 0.0
        while i < self.epochs:
            print("Epoch:",i)
            i += 1
            start_t = int(time.time() * 1000)
            # break
            for j, (batch, _) in enumerate(train_loader):
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                B, L, M = batch.shape
                batch = batch.reshape(B, patch_size, L//patch_size, M)
                batch = batch.permute(0, 3, 2, 1)
                
                self.optimizer.zero_grad()
                loss = self.loss(
                    batch, self.encoder
                )
                loss.backward()
                self.optimizer.step()

            
            end_t = int(time.time() * 1000)
            train_t = float(end_t - start_t)
            train_total_t += train_t
            total_loss = []
            self.encoder.eval()
            with torch.no_grad():
                for j, (batch, _) in enumerate(train_loader):
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    B, L, M = batch.shape
                    batch = batch.reshape(B, patch_size, L//patch_size, M)
                    batch = batch.permute(0, 3, 2, 1)
                    
                    loss = self.loss(
                        batch, self.encoder
                    )
                    total_loss.append(loss.detach().cpu())

            loss = np.average(total_loss)
            self.encoder.train()

            if loss < best_loss:
                best_loss = loss
                early_stopper = 0
                print('Save Encoder Model...')
                path = os.path.join(self.checkpoints, setting)
                best_model_path = os.path.join(path, 'feature_checkpoint.pth')
                torch.save(self.encoder.state_dict(), best_model_path)
            else:
                early_stopper += 1
            if early_stopper == 3:
                early_stopper = 0
                break
            
        
        train_average_t = train_total_t / i
        print(f"average feature embedding training time: {train_average_t}ms")

        return self.encoder, train_average_t
    
