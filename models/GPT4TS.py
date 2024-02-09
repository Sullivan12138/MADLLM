from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time

from models.layers import casual_cnn, triple_loss, prompt

class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    @param dataset Numpy array representing the dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]
    

class FeatureEncoder(nn.Module):
    def __init__(self, batch_size=10, negative_penalty=1,
                 nb_steps=1500, lr=0.001, nb_random_samples=10, in_channels=55,
                 channels=96, depth=10, 
                 reduced_size=384, out_channels=768, kernel_size=3, cuda=False, gpu=0):
        super(FeatureEncoder, self).__init__()
        self.batch_size = batch_size
        self.cuda = cuda
        self.gpu = gpu
        self.nb_steps = nb_steps
        self.loss = triple_loss.TripleLoss(nb_random_samples, negative_penalty)
        self.encoder = casual_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels, kernel_size
        )
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.encoder.double()


    def forward(self, x):
        # x: (batch_size x length x nvars)
        x = x.permute(0, 2, 1)
        x = self.encoder(x) # (batch_size x d_model x length)
        x = x.permute(0, 2, 1) # (batch_size x length x d_model)

        enc_out = rearrange(x, 'b (n s) m -> b n s m', s=25)
        means = enc_out.mean(2, keepdim=True).detach()
        enc_out = enc_out - means
        stdev = torch.sqrt(
            torch.var(enc_out, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
        enc_out /= stdev
        enc_out = rearrange(enc_out, 'b n s m -> b (n s) m')  # (batch_size, length, d_model)

        return enc_out
        

    def fit(self, train_loader):
        # X: (train_size x nvars x num_patch x patch_length)
        print("Start train feature encoder...")

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            print("Epoch:",epochs)
            # break
            for j, (batch, _) in enumerate(train_loader):
                # batch: (batch size x nvars x num_patch x patch_length)
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                B, L, M = batch.shape
                batch = batch.reshape(B, 25, L//25, M)
                batch = batch.permute(0, 3, 2, 1)
                
                self.optimizer.zero_grad()
                loss = self.loss(
                    batch, self.encoder
                )
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            
        print('Save Encoder Model...')
        best_model_path = 'feature_checkpoint.pth'
        torch.save(self.encoder.state_dict(), best_model_path)

        return self.encoder
    

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.patch_num = (configs.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.configs = configs

        # print(self.gpt2)
        
        for _, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        if configs.use_feature_embedding:
            self.feature_embedding = FeatureEncoder(in_channels=configs.enc_in, cuda=True, gpu=0)
            
        if configs.use_prompt_embedding:
            self.prompt_layer = prompt.Prompt(embed_dim=configs.enc_in, top_k=configs.top_k, length=configs.prompt_len, pool_size=configs.pool_size)
            raw_input_len = configs.seq_len + configs.top_k * configs.prompt_len
            self.input_embedding = nn.Linear(raw_input_len,configs.seq_len)

        self.ln_proj = nn.LayerNorm(configs.d_ff)
        self.out_layer = nn.Linear(
            configs.d_ff, 
            configs.c_out, 
            bias=True)
        

    def forward(self, x_enc):
        dec_out = self.anomaly_detection(x_enc) 
        # print("dec_out.shape:",dec_out.shape)
        return dec_out  # [B, L, D]


    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer

        patch_size = self.patch_size
        patch_num = L // patch_size

        if self.configs.use_feature_embedding:
            feature_embedding = self.feature_embedding(x_enc) # (batch_size, length, d_model)

        if self.configs.use_skip_embedding:
            skip_embedding = x_enc.reshape(B, patch_size, patch_num, M)
            skip_embedding = skip_embedding.permute(0,2,1,3)
            skip_embedding = skip_embedding.reshape(B, L, M)
            x_enc = x_enc + skip_embedding

        if self.configs.use_prompt_embedding:
            output = self.prompt_layer(x_enc)
            prompt_prefix = output['prompted_embedding']  # (batch_size, prompt_len, d_model)

        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=patch_size)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')  # (batch_size, length, d_model)

        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))

        if self.configs.use_feature_embedding:
            enc_out = enc_out + feature_embedding
        
        if self.configs.use_prompt_embedding:
            prompt_prefix = rearrange(prompt_prefix, 'b (n s) m -> b n s m', s=patch_size)
            prompt_means = prompt_prefix.mean(2, keepdim=True).detach()
            prompt_prefix = prompt_prefix - prompt_means
            prompt_stdev = torch.sqrt(
                torch.var(prompt_prefix, dim=2, keepdim=True, unbiased=False) + 1e-5)
            prompt_prefix /= prompt_stdev
            prompt_prefix = rearrange(prompt_prefix, 'b n s m -> b (n s) m')  # (batch_size, length, d_model)

            prompt_prefix = torch.nn.functional.pad(prompt_prefix, (0, 768-prompt_prefix.shape[-1]))
            enc_out = torch.cat([prompt_prefix, enc_out], dim=1 )
            enc_out = enc_out.permute(0, 2, 1)
            enc_out = self.input_embedding(enc_out)
            enc_out = enc_out.permute(0, 2, 1)

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state

        dec_out = dec_out[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(dec_out)

        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=patch_size)
        # print(dec_out.shape)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, patch_size, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, patch_size, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')

        return dec_out

    
    def fit_feature(self, x):
        # x: (train_size x nvars x num_patch x patch_length)
        if self.configs.use_feature_embedding:
            self.feature_embedding.fit(x)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'feature_embedding' in name:
                param.requires_grad = False
