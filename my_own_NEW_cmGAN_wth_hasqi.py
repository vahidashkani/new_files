#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:46:59 2023

@author: nca
"""

import torch 
from torchvision.utils import save_image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
import torchaudio
import random
import torch.nn as nn
from joblib import Parallel, delayed
from pesq import pesq
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.utils.data
from utils import *
from torchinfo import summary
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import logging
import torch.multiprocessing as mp
from pystoi import stoi
from torch.nn import init
from hasqi_v2 import hasqi_v2

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
torch.cuda.set_per_process_memory_fraction
#torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<9000>"
allocated_memory = torch.cuda.memory_allocated()
cached_memory = torch.cuda.memory_cached()
print(f"Allocated GPU memory: {allocated_memory / 1024**3:.2f} GB")
print(f"Cached GPU memory: {cached_memory / 1024**3:.2f} GB")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
'''TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
#IMAGE_SIZE = 256
CHANNELS_IMG = 1
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "/home/nca/disc.pth.tar"
CHECKPOINT_GEN = "/home/nca/gen.pth.tar"
'''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),}
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)

def power_uncompress2(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    #b = torch.stack([real_compress, imag_compress], -1)
    return torch.complex(real_compress, imag_compress). squeeze(1)

def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def hasqi_loss(clean, noisy, sr=16000):
    try:
        hasqi_score = hasqi_v2(clean, sr, noisy, sr)
    except:
        # error can happen due to silent period
        hasqi_score = -1
    return hasqi_score


def batch_hasqi(clean, noisy):
    hasqi_score = Parallel(n_jobs=-1)(
        delayed(hasqi_loss)(c, n) for c, n in zip(clean, noisy))
    hasqi_score = np.array(hasqi_score)
    
    # Scale the values from their original range to (0, 1)
    min_value = -1
    max_value = 1  # Adjust this value based on the actual range of hasqi_score
    if -1 in hasqi_score:
        return None
    # Apply scaling
    hasqi_score = (hasqi_score - min_value) / (max_value - min_value)
    # Ensure the values are within the (0, 1) range
    hasqi_score = np.clip(hasqi_score, 0, 1)
    # Convert to a PyTorch tensor on the "cuda" device
    return torch.FloatTensor(hasqi_score).to("cuda")



'''def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    #print("ssssssssssssssssssssssss=  ", pesq_score)
    return torch.FloatTensor(pesq_score).to("cuda")'''
    

class Discriminator(nn.Module):
    def __init__(self, ndf, ratio = 8, in_channel=2):
        super(Discriminator, self).__init__()
        # Layer 1
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (3, 3), (1, 1), (1, 1), bias=False))
        self.norm0 = nn.InstanceNorm2d(ndf, affine=True)
        # Layer 2
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (3, 3), (1, 1), (1, 1), bias=False))
        #self.multiplication1 = torch.matmul(self.conv1, self.conv2)
        self.norm1 = nn.InstanceNorm2d(ndf, affine=True)
        #self.prelu1 = nn.PReLU(ndf)
        self.softmax = nn.Softmax(dim=1)
        # Layer 3
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (3, 3), (1, 1), (1, 1), bias=False))
        self.norm2 = nn.InstanceNorm2d(ndf, affine=True)
        self.prelu2 = nn.PReLU(ndf)
        #self.multiplication2 = torch.matmul(self.softmax, self.prelu2)

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Sqeeze excitation part
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // ratio, in_channel, bias=False),
            nn.Sigmoid())


        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(20, ndf * 2, (5, 5), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (5, 5), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (5, 5), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),)
        
        
        
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        #print("xy = ", xy.shape)
        conv1 = self.conv1(xy)
        norm0 =self.norm0(conv1)
        conv2 = self.conv2(xy)
        norm1 =self.norm1(conv2)
        multipl1 = norm0 * norm1
        #norm1 = self.norm1(multipl1)
        #prelu1 = self.prelu1(norm1)
        softmax = self.softmax(multipl1)
        
        conv3 = self.conv3(xy)
        norm2 = self.norm2(conv3)
        prelu2 = self.prelu2(norm2)
        multipl2 = softmax * prelu2
        #print("multiple2 = ", multipl2.shape)
        #print("xy = ", xy.shape)
        combined_tensor = torch.cat((multipl2, xy), dim=1)
        #print("combined_tensor = ", combined_tensor.shape)

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #Squeeze excitation part
        bs, c, _, _ = xy.shape
        y = self.squeeze(xy).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        #d = xy * y.expand_as(xy)
        d = xy * y
        #print("d = ", d.shape)
        d_weighted = d * 0.3
        combined_tensor_weighted = combined_tensor * 0.3
        concate = torch.cat((combined_tensor, d), dim=1)
        
        #print("d3_weighted = ", d3_weighted)
        #print("multipl2_weighted = ", multipl2_weighted)
        
        return self.layers(concate)

    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),)
        
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),)

    def forward(self, x):
        return self.net(x)


# Conformer Block
# Conformer Block
class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out'''




class DilatedDenseNet(nn.Module):
    def __init__(self, ndf=32, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.in_channels = in_channels
        
        # Layer 1
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, ndf, (3, 3), (1, 1), (1, 1), bias=False))
        self.norm1 = nn.InstanceNorm2d(ndf, affine=True)
        self.prelu1 = nn.PReLU(ndf)
       
        # Layer 2
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels, ndf, (3, 3), (1, 1), (1, 1), bias=False))
        self.norm2 = nn.InstanceNorm2d(ndf, affine=True)
        self.prelu2 = nn.PReLU(ndf)
        
        # Layer 3
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, (3, 3), (1, 1), (1, 1), bias=False))
        self.norm3 = nn.InstanceNorm2d(ndf * 4, affine=True)
        self.prelu3 = nn.PReLU(ndf * 4)
        
        # Layer 4
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 8, (3, 3), (1, 1), (1, 1), bias=False))
        self.norm4 = nn.InstanceNorm2d(ndf * 8, affine=True)
        self.prelu4 = nn.PReLU(ndf * 8)
        
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(384, 64, (3, 3), (1, 1), (1, 1), bias=False))
        self.norm5 = nn.InstanceNorm2d(64, affine=True)
        self.prelu5 = nn.PReLU(64)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, xy):
        conv1 = self.conv1(xy)
        norm1 =self.norm1(conv1)
        prelu1 = self.prelu1(norm1)
        #print("PRELU1 = ", prelu1.shape)
        conv2 = self.conv2(xy)
        norm2 =self.norm2(conv2)
        prelu2 = self.prelu2(norm2)
        #print("PRELU2 = ", prelu2.shape)

        combined_tensor1 = torch.cat((prelu1, prelu2), dim=1)

        softmax1 = self.softmax(combined_tensor1)
        multipl1 = softmax1 * xy
        #print("multipl1 = ", multipl1.shape)

        conv3 = self.conv3(multipl1)
        norm3 = self.norm3(conv3)
        prelu3 = self.prelu3(norm3)
        #print("PRELU3 = ", prelu3.shape)

        conv4 = self.conv4(prelu2)
        norm4 = self.norm4(conv4)
        prelu4 = self.prelu4(norm4)
        #print("PRELU4 = ", prelu4.shape)
        
        # Pad prelu3 with zeros along dimension 1
        combined_tensor2 = torch.cat((prelu4, prelu3), dim=1)
        

        #print("combined_tensor2 = ", combined_tensor2.shape)
        
        conv5 = self.conv5(combined_tensor2)
        norm5 = self.norm5(conv5)
        prelu5 = self.prelu5(norm5)
        #print("PRELU5 = ", prelu5.shape)
        
        return prelu5


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
        nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
        nn.InstanceNorm2d(channels, affine=True),
        nn.PReLU(channels),)
        
        self.dilated_dense = DilatedDenseNet(ndf=32, in_channels=64)
        #self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        
       
    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t1 = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t1) + x_t1
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        ##########################################
        ##########################################
        x_f2 = x_in.permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        ##########################################
        ##########################################
        x_f = self.freq_conformer(x_f) + x_f + x_f2
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f



class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(ndf=32, in_channels=64)
        #self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(ndf=32, in_channels=64)
        #self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


'''class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        ).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)

        mask = self.mask_decoder(out_1)
        out_mag = mask * mag

        complex_out = self.complex_decoder(out_1)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
        
        return final_real, final_imag'''

class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)

        self.mask_decoder = MaskDecoder(
            num_features, num_channel=num_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        noisy_phase = torch.angle(
            torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask = self.mask_decoder(out_5)
        out_mag = mask * mag

        complex_out = self.complex_decoder(out_5)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%













#%%%%%%%%%%%%%%
class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]

        return clean_ds, noisy_ds, length


def load_data(ds_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")  # in linux
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = DemandDataset(train_dir, cut_len)
    test_ds = DemandDataset(test_dir, cut_len)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=n_cpu,)

    return train_dataset, test_dataset


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
epochs=120
batch_size=4
log_interval=500
decay_epoch=30
init_lr=5e-4
cut_len=16000*2
data_dir = os.getcwd()
#data_dir = "/home/nca/Microsoft_challenge_codes/CMGAN-main/"
#save_model_dir = "/home/nca/Microsoft_challenge_codes/CMGAN-main/My_own_models/with_HASQI_loss/"
save_model_dir = data_dir 
#print("sssssssssssssssssssssssssssssssssssssssssssssssss=  ", save_model_dir + '/')
loss_weights = [0.1, 0.9, 0.2, 0.05]

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
#%%%%%%%%%%%%%%

class Trainer:
    discriminator = Discriminator(ndf=16).to(DEVICE)
    def __init__(self, train_ds, test_ds):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        #self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(DEVICE)
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(DEVICE)
        summary(
            self.model, [(1, 2, cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
        )
        self.discriminator = Discriminator(ndf=16).cuda()
        summary(
            self.discriminator,
            [
                (1, 1, int(self.n_fft / 2) + 1, cut_len // self.hop + 1),
                (1, 1, int(self.n_fft / 2) + 1, cut_len // self.hop + 1),
            ],
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * init_lr
        )
        self.model = DDP(self.model, device_ids=[DEVICE])
        self.discriminator = DDP(self.discriminator, device_ids=[DEVICE])
        self.gpu_id = DEVICE

    def forward_generator_step(self, clean, noisy):
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(DEVICE),
            onesided=True, return_complex=False)
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(DEVICE),
            onesided=True, return_complex=False)
        
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress2(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(DEVICE),
            onesided=True,)

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def calculate_generator_loss(self, generator_outputs):

        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            loss_weights[0] * loss_ri
            + loss_weights[1] * loss_mag
            + loss_weights[2] * time_loss
            + loss_weights[3] * gen_loss_GAN
        )

        return loss

    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = batch_hasqi(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["est_mag"].detach())
            predict_max_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["clean_mag"]
            )
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score) # + F.mse_loss(predict_enhance_metric.flatten(), stoi_score
        else:
            discrim_loss_metric = None

        return discrim_loss_metric

    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(DEVICE)
        noisy = batch[1].to(DEVICE)
        one_labels = torch.ones(batch_size).to(DEVICE)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,)
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Loss: {loss.item()}")
        
        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    def test_step(self, batch):

        clean = batch[0].to(DEVICE)
        noisy = batch[1].to(DEVICE)
        one_labels = torch.ones(batch_size).to(DEVICE)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,)
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])
        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(DEVICE, gen_loss_avg, disc_loss_avg))
        return gen_loss_avg
    
    
    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=decay_epoch, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=decay_epoch, gamma=0.5)
        for epoch in range(epochs):
            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                if (step % log_interval) == 0:
                    logging.info(template.format(self.gpu_id, epoch, step, loss, disc_loss))
            print(f"Epoch {epoch}, Generator Loss: {loss}, Discriminator Loss: {disc_loss}")
            gen_loss = self.test()
            path = os.path.join(save_model_dir,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],)
            path2 =os.path.join(save_model_dir+'my_checkpoint.pth')
            #if not os.path.exists(save_model_dir):
                #os.makedirs(save_model_dir)
            #if DEVICE == 0:
            torch.save(self.model.module.state_dict(), path)
            #torch.save(self.model.state_dict(), "/home/nca/Microsoft_challenge_codes/CMGAN-main/My_own_models/with_HASQI_loss/my_model.pth")
                
            scheduler_G.step()
            scheduler_D.step()


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    if rank == 0:
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(available_gpus)
    train_ds, test_ds = load_data(data_dir, batch_size, 2, cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    #main()
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    print(world_size)
    main(rank= 0, world_size=world_size)
    
    
    