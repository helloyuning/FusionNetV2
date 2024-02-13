# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import ContextBlock
from mmcv.utils.parrots_wrapper import _BatchNorm
import numpy as np
from .builder import BACKBONES
from .base_backbone import BaseBackbone
import matplotlib.pyplot as plt
import cv2
import os
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict
from pylab import *#可视化
import copy
from typing import Optional, List

#TC Block
from functools import partial
import math
from token_cluster.tcformer_layers import Block, TCBlock, OverlapPatchEmbed, CTM
from token_cluster.tcformer_utils import (
    load_checkpoint, get_root_logger, token2map,
    token2map_flops, map2token_flops, cluster_and_merge_flops,
    downup_flops, sra_flops)
from token_cluster.transformer_utils import DropPath, to_2tuple, trunc_normal_
from token_cluster.transformer_utils import trunc_normal_
from token_cluster.tcformer_layers import TCAttention
# from token_cluster.tcformer_utils import DropPath

logger = logging.getLogger(__name__)


BN_MOMENTUM = 0.1


class TCBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=True):
        super().__init__()
        self.t = dim
        self.norm1 = norm_layer(int(dim/2))#256
        self.attn = TCAttention(
            int(dim/2),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, use_sr_layer=use_sr_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs
        else:
            q_dict, kv_dict = inputs, None
        # print("分裂成功")
        x = q_dict['x']
        # print("norm维度",self.t)
        #形状对比 tcf[16, 1024, 128], fusionnet[16,256,512]
        # norm1
        q_dict['x'] = self.norm1(q_dict['x'])
        if kv_dict is None:
            kv_dict = q_dict
        else:
            kv_dict['x'] = self.norm1(kv_dict['x'])

        # attn
        # print("特征后的维度:",self.drop_path(self.attn(q_dict, kv_dict)).shape)
        x = x + self.drop_path(self.attn(q_dict, kv_dict))# [16, 256, 256]->[BS,HW,C]

        # # mlp
        # q_dict['x'] = self.norm2(x)
        # x = x + self.drop_path(self.mlp(q_dict))
        # q_dict['x'] = x

        return x
        # return q_dict

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    #官方代码来源
    """vit: https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)




class ViPNAS_Bottleneck(nn.Module):
    """Bottleneck block for ViPNAS_ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        kernel_size (int): kernel size of conv2 searched in ViPANS.
        groups (int): group number of conv2 searched in ViPNAS.
        attention (bool): whether to use attention module in the end of
            the block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 kernel_size=3,
                 groups=1,
                 attention=False):

        #我家的
        # outputs = []

        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=kernel_size,
            stride=self.conv2_stride,
            padding=kernel_size // 2,
            groups=groups,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        if attention:
            self.attention = ContextBlock(out_channels,
                                          max(1.0 / 16, 16.0 / out_channels))
        else:
            self.attention = None

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.attention is not None:
                out = self.attention(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       4 for ``ViPNAS_Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, ViPNAS_Bottleneck):
            expansion = 1
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ViPNAS_ResLayer(nn.Sequential):
    """ViPNAS_ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ViPNAS ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
        kernel_size (int): Kernel Size of the corresponding convolution layer
            searched in the block.
        groups (int): Group number of the corresponding convolution layer
            searched in the block.
        attention (bool): Whether to use attention module in the end of the
            block.
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 kernel_size=3,
                 groups=1,
                 attention=False,
                 **kwargs):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    kernel_size=kernel_size,
                    groups=groups,
                    attention=attention,
                    **kwargs))
            in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expansion=self.expansion,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        kernel_size=kernel_size,
                        groups=groups,
                        attention=attention,
                        **kwargs))
        else:  # downsample_first=False is for HourglassModule
            for i in range(0, num_blocks - 1):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        expansion=self.expansion,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        kernel_size=kernel_size,
                        groups=groups,
                        attention=attention,
                        **kwargs))
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    kernel_size=kernel_size,
                    groups=groups,
                    attention=attention,
                    **kwargs))

        super().__init__(*layers)


@BACKBONES.register_module()
class fusionNetv2(BaseBackbone):
    """ViPNAS_ResNet backbone.

    "ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"
    More details can be found in the `paper
    <https://arxiv.org/abs/2105.10154>`__ .

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        wid (list(int)): Searched width config for each stage.
        expan (list(int)): Searched expansion ratio config for each stage.
        dep (list(int)): Searched depth config for each stage.
        ks (list(int)): Searched kernel size config for each stage.
        group (list(int)): Searched group number config for each stage.
        att (list(bool)): Searched attention config for each stage.
    """

    arch_settings = {
        34: ViPNAS_Bottleneck,
        50: ViPNAS_Bottleneck,
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 wid=[48, 80, 160, 304, 608],
                 expan=[None, 1, 1, 1, 1],
                 dep=[None, 4, 6, 7, 3],
                 ks=[7, 3, 5, 5, 5],
                 group=[None, 16, 16, 16, 16],
                 att=[None, True, False, True, True],#None, True, False, True, True
                 embed_dims=[256, 512],num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,k=5,
            norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], sample_ratios=[0.25]):
        # None, True, False, True, True原来的att 参数
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = dep[0]
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block = self.arch_settings[depth]
        self.stage_blocks = dep[1:1 + num_stages]

        self._make_stem_layer(in_channels, wid[0], ks[0])

        self.res_layers = []
        _in_channels = wid[0]

        # from transpose_r
        d_model = 256
        dim_feedforward = 1024
        encoder_layers_num = 3#1一开始为三
        n_head = 8
        pos_embedding_type = 'sine'#sine
        w, h = 256, 256

        #原来的trans的pe
        self.ori_pos = nn.Parameter(torch.zeros(1024,1,256))

        self.reduce = nn.Conv2d(128, d_model, 1, bias=False)
        self._make_position_embedding(w, h, d_model, pos_embedding_type)
        self.l2_reshape = nn.Conv2d(160, 128, 1, stride=1, bias=False) #我家的
        self.down = nn.Conv2d(d_model, 304, 1, stride=2, bias=False)  # 我家的
        self.reshape = nn.Conv2d(304 * 2, 304, 1, stride=1, bias=False)  # 我家的
        ##其他层
        self.clu_reshape = nn.Conv2d(d_model, 304, 1, stride=1, bias=False)#cluster reshape
        self.ori_pos1 = nn.Parameter(torch.zeros(4096, 1, 160))
        self.l1_reshape = nn.Conv2d(80, 160, 1, stride=1, bias=False)  # 我家的

        self.FINAL_CONV_KERNEL = 1

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,#1024
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
            return_atten_map=False
        )
        # transfomer cluster
        cur = 0
        self.grid_stride = sr_ratios[0]
        self.sr_ratios = sr_ratios
        dpr = [x.item() for x in torch.linspace(0, 0., sum(depths))]  # stochastic depth decay rule
        i = 0
        # patch_embed = OverlapPatchEmbed(img_size=w if i == 0 else w // (2 ** (i + 1)),
        #                                 patch_size=7 if i == 0 else 3,
        #                                 stride=4 if i == 0 else 2,
        #                                 in_chans=1024 if i == 0 else embed_dims[i - 1],
        #                                 embed_dim=embed_dims[i])

        # block = nn.ModuleList([Block(
        #     dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[i])
        #     for j in range(depths[i])])
        norm = norm_layer(embed_dims[i])
        ctm = CTM(sample_ratios[i], embed_dims[0], embed_dims[0], k)
        # setattr(self, f"patch_embed{i + 1}", patch_embed)
        # setattr(self, f"block{i + 1}", block)
        setattr(self, f"norm{i + 1}", norm)
        setattr(self, f"ctm{i}", ctm)
        # cur += depths[i]

        #TCBlock 求出聚类后的结果
        block2 = nn.ModuleList([TCBlock(
            dim=embed_dims[i+1], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[i])
            for j in range(1)])#只要一层合并特征
        setattr(self, f"block2{i}", block2)

        for i, num_blocks in enumerate(self.stage_blocks):
            expansion = get_expansion(self.block, expan[i + 1])
            _out_channels = wid[i + 1] * expansion
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                kernel_size=ks[i + 1],
                groups=group[i + 1],
                attention=att[i + 1])
            _in_channels = _out_channels
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        """Make a ViPNAS ResLayer."""
        return ViPNAS_ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels, kernel_size):
        """Make stem layer."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 8
                self.pe_w = w // 8
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2*math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward function."""
        # processed = []
        # outputs = []#用于可视化的特征输出
        # names = []#用于可视化的名称赋值
        # cnt = 1
        # x_cpu = x.cpu()
        # outputs.append(x_cpu)
        # names.append('input')
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        #可视化用转换特征为cpu模式
        # x_cpu = x.cpu()
        # outputs.append(x_cpu)
        # names.append('first conv')
        outs = []
        for i, layer_name in enumerate(self.res_layers):

            res_layer = getattr(self, layer_name)
            #print("第", i, "层名", res_layer)
            x = res_layer(x)
            # x_cpu = x.cpu()
            # outputs.append(x_cpu)
            # names.append('stage '+str(i))
            # if i == 0:
            #     #第一层尺寸: [16, 80, 64, 64]
            #     print("第一层的尺寸:", x.shape)
            #     xr_1 = self.l1_reshape(x)
            #     # xr_1 = x
            #     bs, c, h, w = xr_1.shape
            #     # print("第一层的尺寸:", bs, c, h, w)
            #     xr_1 = xr_1.flatten(2).permute(2, 0, 1)
            #     xr_1 = self.global_encoder1(xr_1, pos=self.ori_pos1)#原文用的
            #     xr_1 = xr_1.permute(1, 2, 0).contiguous().view(bs, c, h, w)#[bs,256,32,32]
            #     print("xr1形状：",xr_1.shape)

            if i == 1:
                index = 0
                # patch_embed = getattr(self, f"patch_embed{index + 1}")
                # block = getattr(self, f"block{index + 1}")
                # norm = getattr(self, f"norm{index + 1}")

                # 第二层信号输出为[16, 160, 32, 32] -> [16, 128, 32, 32]
                x_l2_reshape = self.l2_reshape(x)
                x_r = self.reduce(x_l2_reshape)  # [16, 256, 32, 32]
                bs, c, h, w = x_r.shape
                # x_r = x_r.flatten(2).permute(0, 2, 1)#[16, 1024, 256]

                ##tcm
                # x, H, W = patch_embed(x)  #
                # print("通过了")
                # # print("当前的形状:",x.shape, H, W)
                # for blk in block:
                #     x = blk(x, H, W)
                # x = norm(x)
                x_r = x_r.flatten(2).permute(2, 0, 1)
                x_r = self.global_encoder(x_r, pos=self.pos_embedding)#原文用的 [1024, 16, 256]
                # B, H, W = x_r.shape  # [16,1024,256], tcformer_emb [16, 4096, 64]
                x_r = x_r.permute(1, 0, 2)

                # print("一开始形状:",x_r.shape)

                # init token dict

                B, N, _ = x_r.shape
                device = x_r.device
                idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
                agg_weight = x_r.new_ones(B, N, 1)
                token_dict = {'x': x_r,
                              'token_num': N,
                              'map_size': [h, w],
                              'init_grid_size': [h, w],
                              'idx_token': idx_token,
                              'agg_weight': agg_weight}
                ctm = getattr(self, f"ctm{index}")# CTM
                # print("token输入前:",token_dict['x'].shape)#[16,1024,256]
                token_dict = ctm(token_dict)#[16, 256, 512]
                # print('TOKEN输出尺寸:',token_dict[0]['x'].shape,token_dict[1]['x'].shape)
                block2 = getattr(self, f"block2{index}")
                for j, blk in enumerate(block2):
                    x_r = blk(token_dict)
                # print("x的形状:",x.shape)
                #[BS, HW, C]
                x_r = x_r.permute(2, 0, 1)#[16, 256,512]
                # print("交换后x的形状:", x_r.shape)
                # print("通过")
                #下面为可用的
                #当前的形状: [1024,16,256] : tcformer_emb [16, 4096, 64]
                # x_r = x_r.flatten(2).permute(2, 0, 1)
                # x_r = self.global_encoder(x_r, pos=self.pos_embedding)#原文用的 [1024, 16, 256]

                # x_r = self.global_encoder(x_r, pos=self.ori_pos)#vit_pe
                #位置编码大小:pos: [1024,1,256]
                # print("hw",h,w)
                # x_r = x_r.permute(1, 2, 0).contiguous().view(bs, c, h, w)#[bs,256,32,32]
                x_r = x_r.permute(1, 2, 0).contiguous().view(bs, c,int(h/2), int(w/2))  # [bs,256,16,16]
                # print("处理完后的形状:",x_r.shape)



            if i == 2:
                #第三层的输出形状: [16, 304, 16, 16]
                #print("第三层输出信号的形状:", x.shape)
                down = self.clu_reshape(x_r)  # 1x1调整特征尺寸 [16, 304, 16, 16]
                ##提取到cpu
                # down_cpu = down.cpu()
                # outputs.append(down_cpu)
                # names.append('trans feature')
                # print("down调整后的特征",down.shape)
                fusion = torch.cat([x, down], 1)  # [16, 608, 16, 16]
                # print("结合后的特征:", fusion.shape)
                x = self.reshape(fusion)  # [16, 608, 16, 16] -> [16, 304, 16, 16]

            if i in self.out_indices:
                outs.append(x)
        #outs储存所有输出的信号
        if len(outs) == 1:
            # print("类型:",outs)
            return outs[0] #[16, 608, 8, 8]

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    # feature_map = img_batch
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    print("当前特征的尺寸:",num_pic)
    row, col = get_row_col(num_pic)
    print(row,col)
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        # axis('off')
        # title('feature_map_{}'.format(i))
    # plt.imshow(feature_map)
    # plt.savefig('feature_map.png')
    plt.show()

    # # 各个特征图按1：1 叠加
    # feature_map_sum = sum(ele for ele in feature_map_combination)
    # plt.imshow(feature_map_sum)
    # plt.savefig("feature_map_sum.png")


class Hook(object):
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self, module, fea_in, fea_out):
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None


def plot_feature(features, idx=0):
    hh = Hook()
    features.register_forward_hook(hh)

    # forward_model(model,False)

    print(hh.module_name)
    print((hh.features_in_hook[0][0].shape))
    print((hh.features_out_hook[0].shape))

    out1 = hh.features_out_hook[0]

    total_ft = out1.shape[1]
    first_item = out1[0].cpu().clone()

    plt.figure(figsize=(20, 17))

    for ftidx in range(total_ft):
        if ftidx > 99:
            break
        ft = first_item[ftidx]
        plt.subplot(10, 10, ftidx + 1)

        plt.axis('off')
        # plt.imshow(ft[ :, :].detach(),cmap='gray')
        plt.imshow(ft[:, :].detach())
