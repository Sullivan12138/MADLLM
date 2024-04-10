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
from models.layers import prompt

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
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.configs = configs
        self.output_transfer = torch.nn.Linear(self.enc_in, self.d_model)

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
            
        if configs.use_prompt_pool:
            self.prompt_layer = prompt.Prompt(embed_dim=configs.d_model, top_k=configs.top_k, length=configs.prompt_len, pool_size=configs.pool_size)
            raw_input_len = configs.seq_len + configs.top_k * configs.prompt_len
            self.input_embedding = nn.Linear(raw_input_len,configs.seq_len)

        self.ln_proj = nn.LayerNorm(configs.d_ff)
        self.out_layer = nn.Linear(
            configs.d_ff, 
            configs.c_out, 
            bias=True)
        

    def forward(self, x_enc, feature_embedding):
        dec_out = self.anomaly_detection(x_enc, feature_embedding) 
        # print("dec_out.shape:",dec_out.shape)
        return dec_out  # [B, L, D]


    def anomaly_detection(self, x_enc, feature_embedding):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer

        patch_size = self.patch_size
        patch_num = L // patch_size

        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=patch_size)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')  # (batch_size, length, d_model)

        if self.configs.use_skip_embedding:
            skip_embedding = x_enc.reshape(B, patch_size, patch_num, M)
            skip_embedding = skip_embedding.permute(0,2,1,3)
            skip_embedding = skip_embedding.reshape(B, L, M)
            x_enc = x_enc + skip_embedding
        

        x_enc = self.output_transfer(x_enc)
        if self.configs.use_feature_embedding:
            x_enc = x_enc + feature_embedding

        if self.configs.use_prompt_pool:
            output = self.prompt_layer(x_enc)
            prompt_prefix = output['prompted_embedding']  # (batch_size, prompt_len, d_model)
            
            # enc_out = enc_out.permute(0, 2, 1)
            # enc_out = self.input_embedding(enc_out)
            # enc_out = enc_out.permute(0, 2, 1)
            x_enc = torch.cat([prompt_prefix, x_enc], dim=1 )

            x_enc = x_enc.permute(0, 2, 1)
            x_enc = self.input_embedding(x_enc)
            x_enc = x_enc.permute(0, 2, 1)

        

        



        # enc_out = torch.nn.functional.pad(x_enc, (0, 768-enc_out.shape[-1]))  # 此处可能未来要改

        # if self.configs.use_feature_embedding:
        #     enc_out = enc_out + feature_embedding
        
        # if self.configs.use_prompt_embedding:
        #     # prompt_prefix = rearrange(prompt_prefix, 'b (n s) m -> b n s m', s=patch_size)
        #     # prompt_means = prompt_prefix.mean(2, keepdim=True).detach()
        #     # prompt_prefix = prompt_prefix - prompt_means
        #     # prompt_stdev = torch.sqrt(
        #     #     torch.var(prompt_prefix, dim=2, keepdim=True, unbiased=False) + 1e-5)
        #     # prompt_prefix /= prompt_stdev
        #     # prompt_prefix = rearrange(prompt_prefix, 'b n s m -> b (n s) m')  # (batch_size, length, d_model)

        #     prompt_prefix = torch.nn.functional.pad(prompt_prefix, (0, 768-prompt_prefix.shape[-1]))
        #     enc_out = torch.cat([prompt_prefix, enc_out], dim=1 )
        #     enc_out = enc_out.permute(0, 2, 1)
        #     enc_out = self.input_embedding(enc_out)
        #     enc_out = enc_out.permute(0, 2, 1)

        # enc_out = rearrange(enc_out, 'b (n s) m -> b n s m', s=patch_size)
        # means = enc_out.mean(2, keepdim=True).detach()
        # enc_out = enc_out - means
        # stdev = torch.sqrt(
        #     torch.var(enc_out, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
        # enc_out /= stdev
        # enc_out = rearrange(enc_out, 'b n s m -> b (n s) m')

        dec_out = self.gpt2(inputs_embeds=x_enc).last_hidden_state

        
        # dec_out = dec_out.permute(0, 2, 1)
        # dec_out = self.input_embedding(dec_out)
        # dec_out = dec_out.permute(0, 2, 1)

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

        # dec_out = dec_out.permute(0, 2, 1)
        # dec_out = self.input_embedding(dec_out)
        # dec_out = dec_out.permute(0, 2, 1)

        
        
        return dec_out

    
    # def fit_feature(self, x):
    #     # x: (train_size x nvars x num_patch x patch_length)
    #     if self.configs.use_feature_embedding:
    #         self.feature_embedding.fit(x, patch_size=self.patch_size)

    #     for _, (name, param) in enumerate(self.gpt2.named_parameters()):
    #         if 'feature_embedding' in name:
    #             param.requires_grad = False
