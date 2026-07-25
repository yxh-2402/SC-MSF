"""SC-MSF model implementation for the corresponding benchmark."""
import sys
import numpy as np
import copy
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
import math
from copy import deepcopy
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

import seaborn as sns

from SC_MSF.modeling.latent_net import CategoricalLatent, kl_q_p
from SC_MSF.modeling.gmm2d import GMM2D
from SC_MSF.modeling.gmm4d import GMM4D
from SC_MSF.modeling.dynamics.integrator import SingleIntegrator
from SC_MSF.layers.loss import cvae_loss, mutual_inf_mc
from SC_MSF.modeling.TCN import TemporalConvNet
from utils.common import PositionalEncoding, ConcatSquashLinear, ConcatTransformerLinear


class ChannelAttention1D(nn.Module):
    """Channel attention for temporal features in (batch, channels, time) format."""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1D, self).__init__()
        hidden_channels = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, hidden_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(hidden_channels, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention1D(nn.Module):
    """Temporal attention shared across feature channels."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM1D(nn.Module):
    """One-dimensional CBAM used by the DTDE enhancement module."""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(in_planes, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


class DEConv1D(nn.Module):
    """Difference-enhanced convolution for temporal motion changes."""

    def __init__(self, dim):
        super(DEConv1D, self).__init__()
        self.conv_cd = nn.Conv1d(dim, dim, 3, padding=1, bias=True)
        self.conv_fd = nn.Conv1d(dim, dim, 3, padding=1, bias=True)
        self.conv_bd = nn.Conv1d(dim, dim, 3, padding=1, bias=True)
        self.conv_normal = nn.Conv1d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        out_cd = self.conv_cd(x)
        x_fd = F.pad(x[:, :, 1:], (0, 1)) - x
        out_fd = self.conv_fd(x_fd)
        x_bd = x - F.pad(x[:, :, :-1], (1, 0))
        out_bd = self.conv_bd(x_bd)
        out_normal = self.conv_normal(x)
        return out_cd + out_fd + out_bd + out_normal


class DTDEConv1D(nn.Module):
    """Difference, trend/detail decomposition, and attention enhancement."""

    def __init__(self, in_channels, kernel_size=3, use_wavelet=True):
        super(DTDEConv1D, self).__init__()
        self.in_channels = in_channels
        self.use_wavelet = use_wavelet
        self.deconv = DEConv1D(in_channels)
        self.cbam = CBAM1D(in_channels, ratio=max(in_channels // 4, 1))

        if use_wavelet:
            self.low_freq = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm1d(in_channels),
            )
            self.high_enhance = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 1),
                nn.ReLU(),
                nn.Conv1d(in_channels, in_channels, 1),
                nn.Sigmoid(),
            )

        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.ReLU(),
        )

        if use_wavelet:
            nn.init.constant_(self.low_freq[0].weight, 1.0 / kernel_size)

    def forward(self, x):
        identity = x
        x_deconv = self.deconv(x)

        if self.use_wavelet:
            x_low = self.low_freq(x)
            x_high = x - x_low
            x_high = x_high * self.high_enhance(x_high)
            x_wavelet = x_low + x_high
        else:
            x_wavelet = x

        x_fused = x_deconv + x_wavelet
        x_attn = self.cbam(x_fused)
        return self.fusion(x_attn) + identity


def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)
    p_attn = softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query = query.reshape(nbatches ,1,1,256)
        # key = key.reshape(nbatches ,1,1,256)
        # value = value.reshape(nbatches ,1,1,256)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # print('q为',query.shape,'k为',key.shape,'v为',value.shape)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        A = self.linears[-1](x)
        return self.linears[-1](x)


class CrossAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(CrossAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        y = query
        batch_size = y.size(0)
        query = query.reshape(nbatches, 1, 1, 256)
        key = key.reshape(nbatches, 1, 1, 256)
        value = value.reshape(nbatches, 1, 1, 256)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        a = self.linears[-1](x).reshape(batch_size, 256)
        return a + y
        # return a


import torch
import torch.nn as nn


class PIE(nn.Module):
    def __init__(self, cfg, dataset_name=None):
        super(PIE, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.K = self.cfg.K
        self.pose_channel = 32
        self.flow_channel = 32
        # print(self.flow_channel)
        # self.K = 1
        # print(self.K)
        self.param_scheduler = None
        # encoder
        self.box_embed = nn.Sequential(nn.Linear(self.cfg.GLOBAL_INPUT_DIM, self.cfg.INPUT_EMBED_SIZE),
                                       nn.ReLU())
        self.box_encoder = nn.GRU(input_size=self.cfg.INPUT_EMBED_SIZE,
                                  hidden_size=self.cfg.ENC_HIDDEN_SIZE,
                                  batch_first=True)
        self.ped_flow_emb = nn.Linear(2, 256)
        # Match the JAAD innovation path while retaining PIE's 2-D pedestrian-flow input.
        self.flow_dtde = DTDEConv1D(in_channels=256, kernel_size=3, use_wavelet=True)
        # self.traj_speed_emb = nn.Linear(256 * 2, 256)
        # encoder for future trajectory

        self.node_future_encoder_h = nn.Linear(self.cfg.GLOBAL_INPUT_DIM, 32)
        self.gt_goal_encoder = nn.GRU(input_size=self.cfg.DEC_OUTPUT_DIM,
                                      hidden_size=32,
                                      bidirectional=True,
                                      batch_first=True)

        self.hidden_size = self.cfg.ENC_HIDDEN_SIZE
        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_size,
                                             128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.cfg.LATENT_DIM * 2))
        # posterior
        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.GOAL_HIDDEN_SIZE,
                                              128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.LATENT_DIM * 2))

        #  add bidirectional predictor

        self.dec_init_hidden_size = self.hidden_size + self.cfg.LATENT_DIM if self.cfg.DEC_WITH_Z else self.hidden_size

        self.enc_h_to_forward_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size,
                                                          self.cfg.DEC_HIDDEN_SIZE),
                                                nn.ReLU(),
                                                )
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(self.cfg.DEC_HIDDEN_SIZE,
                                                              self.cfg.DEC_INPUT_SIZE),
                                                    nn.ReLU(),
                                                    )
        self.traj_dec_forward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,
                                           hidden_size=self.cfg.DEC_HIDDEN_SIZE)

        self.enc_h_to_back_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size,
                                                       self.cfg.DEC_HIDDEN_SIZE),
                                             nn.ReLU(),
                                             )

        self.traj_dec_input_backward = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM,  # 2 or 4
                                                               self.cfg.DEC_INPUT_SIZE),
                                                     nn.ReLU(),
                                                     )
        self.traj_dec_input_backward2 = nn.Sequential(nn.Linear(256,  # 2 or 4
                                                                self.cfg.DEC_INPUT_SIZE),
                                                      nn.ReLU(),
                                                      )
        self.traj_dec_backward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)

        self.traj_output = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2,  # merged forward and backward
                                     self.cfg.DEC_OUTPUT_DIM)

        self.traj_dec_backward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)
        # goal predictor
        ##########3##########################################################################################################################
        self.goal_decoder = nn.Sequential(nn.Linear(self.dec_init_hidden_size,
                                                    128),
                                          nn.ReLU(),
                                          nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.cfg.DEC_OUTPUT_DIM))
        self.traj_decoder = nn.Sequential(nn.Linear(self.cfg.DEC_HIDDEN_SIZE,
                                                    128),
                                          nn.ReLU(),
                                          nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, self.cfg.DEC_OUTPUT_DIM))

        ###################################################################################### 关键帧编码

        self.key_pose_encoder_rnn = nn.RNN(input_size=36,
                                           hidden_size=32,
                                           batch_first=True)

        ########################################################################################
        ########################################################################################
        # 添加注意力机制

        self.attn_h = MultiHeadAttention(1, 256)
        self.cattn = CrossAttention(1, 256)
        self.flow_att = nn.Sequential(nn.Linear(32, 256), nn.ReLU())
        self.pose_att = nn.Sequential(nn.Linear(32, 256), nn.ReLU())

        #######################################################################################
        # 光流编码
        self.cur_flow_cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            # nn.AvgPool2d(4, 4),
            nn.AvgPool2d(2, 2),
            # nn.Linear(64,32)
        )
        self.cur_flow_cnnz_enc = nn.Sequential(

            nn.Linear(24, 24),
            nn.ReLU())
        self.flow_seq_rnn = nn.RNN(input_size=24,
                                   hidden_size=32,
                                   batch_first=True)

        self.flow_seq_enc = nn.Sequential(
            nn.Linear(448, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.flow_channel),
            nn.ReLU()
        )
        # 图像体特征
        # self.cur_img_cnn = nn.Sequential(
        #      # The convolutional layers.
        #     # The first convolutional layer.
        #     nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        #     nn.Sigmoid(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # The second convolutional layer.
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        #     nn.Sigmoid(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        self.fc = nn.Sequential(  # The fully connected layer
            nn.Linear(1000, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 32)
        )

        # Scene Context Memory: scene-guided multimodal attention and FiLM modulation.
        self.scene_encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
        )
        self.scene_guided_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.scene_gamma = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )
        self.scene_beta = nn.Linear(256, 256)
        self.scene_fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        # self.cnn1d = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=1, kernel_size=8),
        # nn.Sigmoid(),
        # nn.AvgPool1d(2),
        # nn.Linear(496,256),
        # nn.Sigmoid()

        # )
        # self.img = nn.Sequential(  # The fully connected layer
        #     nn.Linear(256 + 32, 256),
        #     nn.Sigmoid())
        #######################################################
        num_channels = [1, 1, 1, 1]
        # self.tcn_x = TemporalConvNet(15, num_channels, kernel_size=8, dropout=0.1)
        self.tcn_p = TemporalConvNet(4, num_channels, kernel_size=8, dropout=0.1)
        self.tcn_f = TemporalConvNet(15, num_channels, kernel_size=8, dropout=0.1)
        self.tcn_t = TemporalConvNet(15, num_channels, kernel_size=8, dropout=0.1)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.init_weights()

        ######################################################################################
        # 数据质量评价
        self.p = nn.Sequential(
            nn.Linear(36, 256), nn.Sigmoid(),
        )
        self.f = nn.Sequential(
            nn.Linear(256, 256), nn.Sigmoid(),
        )
        self.t = nn.Sequential(
            nn.Linear(4, 256), nn.Sigmoid(),
        )
        # self.t = nn.Sequential(
        #     nn.Linear(4, 64), nn.ReLU(),
        #     nn.Linear(64, 128), nn.ReLU(),
        #     nn.Linear(128, 256), nn.Sigmoid(),
        # )
        # self.t = nn.Sequential( nn.Linear(60, 32), nn.ReLU(),
        #                         nn.Linear(32, 16), nn.ReLU(),
        #                         nn.Linear(16, 1), nn.ReLU ())
        # self.p = nn.Sequential(
        #                         nn.Linear(144, 64), nn.ReLU(),
        #                         nn.Linear(64, 32), nn.ReLU(),
        #                         nn.Linear(32, 1), nn.ReLU())
        # self.f = nn.Sequential(
        #ped_flow_emb
        #     nn.Linear(5376, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1),
        #     nn.ReLU()
        # )
        ####################################################################
        context_dim = 256
        self.pos_emb = PositionalEncoding(d_model=context_dim, dropout=0.1, max_len=45)
        self.layer = nn.TransformerEncoderLayer(d_model=context_dim, nhead=2, dim_feedforward=1 * context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=1)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=2, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        ########################################################################
        self.hxe = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 20),
                                 nn.Tanh(),
                                 )
        self.Softmax = nn.Softmax(dim=1)

        self.vol = nn.Sequential(nn.Linear(1, 16), nn.Sigmoid(), )
        self.ang = nn.Sequential(nn.Linear(1, 16), nn.Sigmoid(), )
        self.vol_e = nn.GRU(input_size=16,
                            hidden_size=16,
                            batch_first=True)
        self.ang_e = nn.GRU(input_size=16,
                            hidden_size=16,
                            batch_first=True)


    #########################################################################################################################
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def gaussian_latent_net(self, enc_h, h_x, cur_state, target=None, z_mode=None):
        # get mu, sigma
        # 1. sample z from piror
        z_mu_logvar_p = self.p_z_x(enc_h)
        # print(z_mu_logvar_p)
        z_mu_p = z_mu_logvar_p[:, :self.cfg.LATENT_DIM]
        z_logvar_p = z_mu_logvar_p[:, self.cfg.LATENT_DIM:]
        if target is not None:
            # 2. sample z from posterior, for training only
            initial_h = self.node_future_encoder_h(cur_state)
            # print(initial_h)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            _, target_h = self.gt_goal_encoder(target, initial_h)
            target_h = target_h.permute(1, 0, 2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])

            target_h = F.dropout(target_h,
                                 p=0.25,
                                 training=self.training)

            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.cfg.LATENT_DIM]
            z_logvar_q = z_mu_logvar_q[:, self.cfg.LATENT_DIM:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_q.exp() / z_logvar_p.exp()) + \
                         (z_mu_p - z_mu_q).pow(2) / z_logvar_p.exp() - \
                         1 + \
                         (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = 0.0

        # 4. Draw sample
        hxe = self.hxe(h_x)
        hx_conf = self.Softmax(hxe)
        # hx_conf =  hxe
        # print(hx_conf)
        # print(h_x)

        data_sort, index = torch.sort(hx_conf, descending=False)
        tail_num = 0

        for data in data_sort:
            for tail in data:
                if tail < 0.055:
                    continue
                else:
                    tail_num = tail_num + 1

        # print(tail_num)
        # print(hx_conf[])
        # print(torch.sum(hx_conf[0]))
        # print(torch.sum(hx_conf[:,0])/ehxe.shape[0] * 20)
        # tail_num_real = int(torch.sum(hx_conf[:,0])/hxe.shape[0] * 20)
        # tail_num_real = 4
        tail_num_real = int(tail_num / hxe.shape[0])
        head_num_real = 20 - tail_num_real
        # print(head_num_real)
        # if enc_h.shape[0] < 128:
        #    print('tail_num_real', tail_num_real, ' head_num_real', head_num_real)
        # print(  data_sort)
        # print(tail_num_real)
        # 头部抽样
        if head_num_real != 0:
            if head_num_real % 2 != 0:  # 奇数
                if head_num_real == 1:
                    K_samples_h = torch.randn(enc_h.shape[0], head_num_real, self.cfg.LATENT_DIM, device=enc_h.device)
                    K_samples_one = torch.ones(enc_h.shape[0], head_num_real, self.cfg.LATENT_DIM, device=enc_h.device)
                    K_samples_head = K_samples_one + K_samples_h
                else:
                    fnum = (head_num_real - 1) / 2
                    znum = fnum + 1
                    K_samples_h = torch.randn(enc_h.shape[0], head_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # *hx
                    K_samples_h = K_samples_h / head_num_real
                    K_samples_one = torch.ones(enc_h.shape[0], head_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # +hx
                    K_sample_head = -K_samples_one
                    for i in range(head_num_real):
                        if i < fnum + 1:
                            K_sample_head[:, i, :] = K_sample_head[:, i, :] - K_sample_head[:, i, :] / fnum * i
                        else:
                            K_sample_head[:, i, :] = K_samples_one[:, i, :] / znum * (i - fnum)
                        # print('K_sample_head',K_sample_head[:, i, :] )
                    K_samples_head = K_sample_head + K_samples_h
            else:
                fnum = (head_num_real) / 2
                znum = fnum
                K_samples_h = torch.randn(enc_h.shape[0], head_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # *hx
                K_samples_h = K_samples_h / head_num_real
                K_samples_one = torch.ones(enc_h.shape[0], head_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # +hx
                K_sample_head = -K_samples_one
                for i in range(head_num_real):
                    if i < fnum + 1:
                        K_sample_head[:, i, :] = K_sample_head[:, i, :] - K_sample_head[:, i, :] / fnum * i
                    else:
                        K_sample_head[:, i, :] = K_samples_one[:, i, :] / znum * (i - fnum)
                    # print('K_sample_head', K_sample_head[:, i, :])
                K_samples_head = K_sample_head + K_samples_h
        else:
            K_samples_head = None

            # 尾部取样
        if tail_num_real != 0:
            if tail_num_real % 2 != 0:
                if tail_num_real == 1:
                    K_samples_t = torch.randn(enc_h.shape[0], tail_num_real, self.cfg.LATENT_DIM, device=enc_h.device)
                    K_samples_one = torch.ones(enc_h.shape[0], tail_num_real, self.cfg.LATENT_DIM, device=enc_h.device)
                    K_samples_tail = -3 * K_samples_one + K_samples_t
                else:
                    fnum = (tail_num_real - 1) / 2
                    znum = fnum + 1
                    K_samples_t = torch.randn(enc_h.shape[0], tail_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # *hx
                    K_samples_t = K_samples_t / tail_num_real
                    K_samples_one = torch.ones(enc_h.shape[0], tail_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # +hx
                    K_sample_tail = -3 * K_samples_one
                    for i in range(tail_num_real):
                        if i < fnum:
                            K_sample_tail[:, i, :] = K_sample_tail[:, i, :] + K_samples_one[:, i, :] * 2 / fnum * i
                        else:
                            K_sample_tail[:, i, :] = K_samples_one[:, i, :] + K_samples_one[:, i, :] / znum * (
                                    i - fnum + 1)
                            # print('K_sample_tail' ,K_sample_tail[:, i, :] )
                    K_samples_tail = K_sample_tail + K_samples_t
            else:
                fnum = (tail_num_real) / 2
                znum = fnum
                K_samples_t = torch.randn(enc_h.shape[0], tail_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # *hx
                K_samples_t = K_samples_t / tail_num_real
                K_samples_one = torch.ones(enc_h.shape[0], tail_num_real, self.cfg.LATENT_DIM, device=enc_h.device)  # +hx
                K_sample_tail = -3 * K_samples_one
                for i in range(tail_num_real):
                    if i < fnum:
                        K_sample_tail[:, i, :] = K_sample_tail[:, i, :] + K_samples_one[:, i, :] * 2 / fnum * i
                    else:
                        K_sample_tail[:, i, :] = K_samples_one[:, i, :] + K_samples_one[:, i, :] / znum * (i - fnum + 1)
                    # print('K_sample_tail', K_sample_tail[:, i, :])
                K_samples_tail = K_sample_tail + K_samples_t
        else:
            K_samples_tail = None

        if head_num_real == 0:
            K_samples = K_samples_tail
        elif tail_num_real == 0:
            K_samples = K_samples_head
        else:
            K_samples = torch.cat([K_samples_head, K_samples_tail], dim=-2)
        # K_samples
        # print(K_samples[0])
        K_samples1 = torch.randn(enc_h.shape[0], self.K, self.cfg.LATENT_DIM)

        # data = K_samples.reshape(1, enc_h.shape[0] * 20 * 32).cpu()

        # data1 = K_samples1.reshape(1, enc_h.shape[0] * 20 * 32).cpu()  # 数组数据

        # n, bins, patches = plt.hist(data, 50, edgecolor='#169acf', linewidth=1, density=True)  # 绘制直方图
        # y = ((1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (1 * (bins)) ** 2))
        # plt.plot(bins, y, '-', color='red', linewidth=2)

        # n = n.astype('int')
        # for i in range(len(patches)):
        #     patches[i].set_facecolor(plt.cm.viridis(n[i] / max(n)))
        # plt.title('正太分布', fontsize=12)
        # plt.xlabel('bins', fontsize=10)
        # plt.ylabel(' Frequency', fontsize=12)
        # plt.legend()
        # plt.show()  #
        # print(K_samples.shape)
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, self.K, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, self.K, 1)

        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD

    def encode_variable_length_seqs(self, original_seqs, key_pose, flow_seq_enc, lower_indices=None, upper_indices=None,
                                    total_length=None):
        '''
        take the input_x, pack it to remove NaN, embed, and run GRU
        '''
        bs, tf = original_seqs.shape[:2]
        if lower_indices is None:
            lower_indices = torch.zeros(bs, dtype=torch.int)
        if upper_indices is None:
            upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
        if total_length is None:
            total_length = max(upper_indices) + 1
        # This is done so that we can just pass in self.prediction_timesteps
        # (which we want to INCLUDE, so this will exclude the next timestep).
        inclusive_break_indices = upper_indices + 1
        pad_list = []
        length_per_batch = []
        for i, seq_len in enumerate(inclusive_break_indices):
            pad_list.append(original_seqs[i, lower_indices[i]:seq_len])
            length_per_batch.append(seq_len - lower_indices[i])

        # 1. embed and convert back to pad_list
        x = self.box_embed(torch.cat(pad_list, dim=0))
        ##

        pad_list = torch.split(x, length_per_batch)

        # 2. run temporal

        packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
        packed_output, h_x = self.box_encoder(packed_seqs)
        # print(packed_output)
        ################################################################################################################

        ##########################################################################################################################

        # pad zeros to the end so that the last non zero value
        output, _ = rnn.pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=total_length)
        # print(output,output.shape) # torch.Size([128, 15, 256])
        # output = torch.cat((output, key_pose_zero), dim=2)  # (1,128,292)
        # output = self.pose_cave(output)
        return output, h_x

    def encoder(self, x, key_pose, flow_seq_enc, first_history_indices=None):
        '''
        x: encoder inputs
        '''

        outputs, _ = self.encode_variable_length_seqs(x, key_pose, flow_seq_enc, lower_indices=first_history_indices)
        outputs = F.dropout(outputs,
                            p=self.cfg.DROPOUT,
                            training=self.training)
        if first_history_indices is not None:
            last_index_per_sequence = -(first_history_indices + 1)
            return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]
        else:
            # if no first_history_indices, all sequences are full length
            return outputs[:, -1, :]

    def forward(self, input_x,
                target_y=None,
                cur_pose=None,
                str_pose=None,
                key_pose=None,
                flow=None,
                key_flow_seq=None,
                pose=None,
                cur_img=None,
                obd_speed=None,
                heading_angle=None,
                obs_cam_flow=None,
                obs_ped_flow=None,
                neighbors_st=None,
                adjacency=None,
                z_mode=False,
                cur_pos=None,

                first_history_indices=None):
        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''

        gt_goal = target_y[:, -1] if target_y is not None else None
        cur_pos = input_x[:, -1, :] if cur_pos is None else cur_pos
        batch_size, seg_len, _ = input_x.shape
        ####自车运动
        # 当 obs_ped_flow 未提供时，从空间光流 flow 均值推导 (B, 15, 2)。
        if obs_ped_flow is None and flow is not None:
            obs_ped_flow = flow.reshape(batch_size, 15, 2, 24, 8).mean(dim=[-2, -1]).to(input_x.device)
        if obs_ped_flow is None:
            raise ValueError("PIE requires obs_ped_flow or a spatial flow tensor.")
        obs_ped_flow = obs_ped_flow.to(input_x.device)
        flow_seq = self.ped_flow_emb(obs_ped_flow)

        # DTDE enhancement follows the JAAD innovation module. Conv1d expects (B, C, T).
        flow_seq = self.flow_dtde(flow_seq.transpose(1, 2)).transpose(1, 2)

        ####################################################################################################################
        ############################################################## 提取骨骼关键帧序列
        device = input_x.device
        # print(cur_pose.shape)
        cur_pose = cur_pose.reshape(batch_size, 36).to(device)  # 首帧
        # print(str_pose.shape)
        str_pose = str_pose.reshape(batch_size, 36).to(device)  # 尾帧
        key_pose = key_pose.reshape(batch_size, 8, 36).to(device)  # 关键帧两帧
        key_pose_seq = torch.zeros((batch_size, 4, 36)).to(device)  # 关键帧序列
        # key_pose_seq[:,0,:] = str_pose
        # key_pose_seq[:, 1, :] = key_pose[:,0,:]
        # key_pose_seq[:, 2, :] = key_pose[:,1,:]
        # key_pose_seq[:, 3, :] = key_pose[:, 2, :]
        # # key_pose_seq[:, 4, :] = key_pose[:, 3, :]
        # key_pose_seq[:, 4, :] = cur_pose
        # key_pose_seq[:, 0, :] = str_pose
        key_pose_seq[:, 0, :] = str_pose
        key_pose_seq[:, 1, :] = key_pose[:, 0, :]
        key_pose_seq[:, 2, :] = key_pose[:, 1, :]
        key_pose_seq[:, 3, :] = cur_pose
        p = self.tcn_p(key_pose_seq.reshape(batch_size, 4, 36))[:, 0, :]
        p = self.p(p)
        # print(p.shape)
        key_pose_seq, key_pose_hx = self.key_pose_encoder_rnn(key_pose_seq)  # 对序列进行rnn编码
        key_pose_seq_cave = key_pose_hx.reshape(batch_size, 32)  # self.key_pose_seq_enc(key_pose_seq)
        ############################################################ 提取光流序列帧
        # print(flow.shape)

        # TCN treats the 15 observed frames as channels and predicts flow quality weights.
        f = self.tcn_f(flow_seq)[:, 0, :]
        f = self.f(f)
        # CrossAttention 需要 (B, 256) 单向量，对 15 帧取均值
        flow_seq_enc = flow_seq.mean(dim=1)
        # print(f.shape)
        t = self.tcn_t(input_x)
        t = self.t(t)
        # print(t.shape)
        #####
        '''
        data = torch.stack([p[0,0],t[0,0,0],f[0,0]], dim=0).unsqueeze(0).cpu().numpy()
        # 使用 seaborn 库的 heatmap 函数生成热力图
        sns.heatmap(data, cmap="YlGnBu")
        plt.show()
        '''
        #####不考虑
        '''
        # f = flow_seq
        flow_seq_zero = torch.zeros((batch_size, 14, 24)).to(device)
        for i in range(14):
            flow_h = torch.floor(flow_seq[:, i, :, 0:8, :])  # (128,2,16,16)
            flow_b = torch.floor(flow_seq[:, i, :, 8:16, :])
            flow_l = torch.floor(flow_seq[:, i, :, 16:24, :])
            flow_h = self.cur_flow_cnn(flow_h).reshape(batch_size, 8)  # cnn编码
            flow_b = self.cur_flow_cnn(flow_b).reshape(batch_size, 8)
            flow_l = self.cur_flow_cnn(flow_l).reshape(batch_size, 8)
            flow_enc = torch.cat((flow_h, flow_b, flow_l), dim=1).to(device)  # 三部分合成
            flow_seq_zero[:, i, :] = flow_enc

        flow_seq_rnn, _ = self.flow_seq_rnn(flow_seq_zero)  # rnn编码
        flow_seq_rnn = flow_seq_rnn.reshape(batch_size, 448)
        flow_seq_enc = self.flow_seq_enc(flow_seq_rnn)
        # print(flow_seq_enc.shape)
        '''
        #########################################################################3图像特征
        obd_speed = obd_speed.to(device)
        heading_angle = heading_angle.to(device)
        # print(obd_speed.shape,heading_angle.shape)
        obd_speed_emb = self.vol(obd_speed)
        _, obd_speed_e = self.vol_e(obd_speed_emb)
        # print(obd_speed_e.shape)
        heading_angle_emb = self.ang(heading_angle)
        _, heading_angle_e = self.ang_e(heading_angle_emb)
        sp_ag = torch.cat((obd_speed_e[0], obd_speed_e[0]), dim=-1).to(device)
        # print(obd_speed_e.shape)
        img_h = self.fc(cur_img.to(device))
        # print(sp_ag.shape, img_h.shape)

        # img_h = img_h - img_h * sp_ag
        # img_h = self.fc(cur_img.to(device))

        # 1. encoder
        # print(self.cfg.FC,key_pose_seq_cave.shape,flow_seq_enc.shape)
        h_x = self.encoder(input_x, key_pose_seq_cave, flow_seq_enc, first_history_indices)
        #############################################################
        h_y = h_x
        flow_seq_enc_y = flow_seq_enc   # 已是 (B, 256)
        key_pose_y = key_pose

        #################################################################
        key_pose = key_pose_seq_cave

        # flow_seq_enc = self.flow_att(flow_seq_enc)
        key_pose = self.pose_att(key_pose)
        key_pose_y = key_pose * p
        key_pose = self.attn_h(key_pose, key_pose, key_pose).reshape(batch_size, 256)
        #flow_seq_enc_y = self.flow_att(flow_seq_enc_y)
        h_p = self.cattn(key_pose_y, flow_seq_enc * f, flow_seq_enc_y * f)
        h_x = self.cattn(h_x * t[:, 0, :], h_p, h_y).reshape(batch_size, 256)
        # Scene Context Memory follows the JAAD innovation path.
        scene_feat = self.scene_encoder(img_h)
        multimodal_seq = torch.stack([h_x, key_pose, flow_seq_enc], dim=1)
        scene_context, _ = self.scene_guided_attn(
            scene_feat.unsqueeze(1),
            multimodal_seq,
            multimodal_seq,
        )
        scene_context = scene_context.squeeze(1)
        gamma = self.scene_gamma(scene_feat)
        beta = self.scene_beta(scene_feat)
        h_x_modulated = gamma * h_x + beta
        h_m = self.scene_fusion(h_x_modulated + 0.1 * scene_context)

        ###############################################################

        # 2-3. latent net and goal decoder
        # print(input_x[:, -1, :].shape,target_y.shape)
        Z, KLD = self.gaussian_latent_net(h_m, h_x, input_x[:, -1, :], target_y, z_mode=False)
        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        ################################################################原方法

        pred_goal = self.goal_decoder(enc_h_and_z)
        dec_h = enc_h_and_z #if self.cfg.DEC_WITH_Z else h_x
        pred_traj = self.pred_future_traj(dec_h, pred_goal)

        ################################################################################################################
        cur_pos = input_x[:, None, -1, :] if cur_pos is None else cur_pos.unsqueeze(1)
        pred_goal = pred_goal + cur_pos
        pred_traj = pred_traj + cur_pos.unsqueeze(1)
        # 5. compute loss
        if target_y is not None:
            # train and val
            loss_goal, loss_traj = cvae_loss(pred_goal,
                                             pred_traj,
                                             target_y,
                                             best_of_many=self.cfg.BEST_OF_MANY
                                             )
            loss_dict = {'loss_goal': loss_goal, 'loss_traj': loss_traj, 'loss_kld': KLD}
        else:
            # test
            loss_dict = {}

        return pred_goal, pred_traj, loss_dict, None, None

    def pred_future_traj(self, dec_h, G):
        '''
        use a bidirectional GRU decoder to plan the path.
        Params:
            dec_h: (Batch, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim)
            G: (Batch, K, pred_dim)
        Returns:
            backward_outputs: (Batch, T, K, pred_dim)
        '''
        pred_len = self.cfg.PRED_LEN

        K = G.shape[1]

        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)
        if len(forward_h.shape) == 2:
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(pred_len):  # the last step is the goal, no need to predict
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)

        forward_outputs = torch.stack(forward_outputs, dim=1)
        # print(forward_outputs.shape)
        final_emb = self.pos_emb(forward_outputs).permute(1, 0, 2)
        forward_outputs = self.transformer_encoder(final_emb).permute(1, 0, 2)

        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(dec_h)
        if len(backward_h.shape) == 2:
           backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])
        backward_input = self.traj_dec_input_backward(G)  # torch.cat([G])
        backward_input = backward_input.view(-1, backward_input.shape[-1])

        for t in range(pred_len - 1, -1, -1):

            backward_h = self.traj_dec_backward(backward_input, backward_h)
            backward_input = self.traj_dec_input_backward2(backward_h)
            backward_outputs.append(backward_h)
        # print(backward_outputs.shape)
        backward_outputs = torch.stack(backward_outputs, dim=1)
        backward_outputs = self.transformer_decoder(backward_outputs, forward_outputs).view(-1, K, 45, 256)
        # inverse because this is backward
        backward_outputs = self.traj_decoder(backward_outputs).permute(0, 2, 1, 3)
        return backward_outputs
