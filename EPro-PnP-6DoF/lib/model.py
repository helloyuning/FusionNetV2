"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import torch
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
import os, sys
from torchsummary import summary

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
import utils.fancy_logger as logger
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

from models.resnet_backbone import ResNetBackboneNet
from models.resnet_rot_head import RotHeadNet
from models.resnet_trans_head import TransHeadNet
from models.CDPN import CDPN
from torchsummary import summary
from models.resnet_test import resnet34

from my_net.vipnas_resnet import ViPNAS_ResNet
from my_net.vipnas_resnet_fusion import ViPNAS_Fusion
from my_net.fusionNetv2 import fusionNetv2
# from my_net.vit import ViT
from my_net.hr_base import HRNET_base
# from my_net.pose_tokenpose_b import TokenPose_B
from my_net.fusion_resnet import ResNetBackboneNet as fusion_res
from my_net.transpose_r import get_pose_net

#FusionNetV2
from my_net.pvt import PyramidVisionTransformer as pvt
from token_cluster.tcformer import tcformer_light
from functools import partial
import torch.nn as nn
# from my_net.linear_FN import linear_FN
from my_net.cluster_edge_FN import cluster_edge
from my_net.res_edge import Res_edge
from my_net.FN_edge import FN_edge
from  my_net.cluster_FN import cluster_FN
# from  my_net.FN_linear import FN_linear
from my_net.FN_fff import FN_fff

# Specification
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}

# my_init_model = '/home/ivclab/path/EPro-PnP-main/EPro-PnP-6DoF/my_init_models/vipnas_res50_coco_256x192-cc43b466_20210624.pth'
# my_init_model = '/home/ivclab/path/EPro-PnP-main/EPro-PnP-6DoF/my_init_models/vitpose+_huge.pth'
mynet_channels = [768,520,304,608]


# Re-init optimizer
def build_model(cfg):
    ## get model and optimizer
    if 'resnet' in cfg.network.arch:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[cfg.network.back_layers_num]#cfg.network.back_layers_num
        # backbone_net = ResNetBackboneNet(block_type, layers, cfg.network.back_input_channel, cfg.network.back_freeze)
        # summary(backbone_net)
        # backbone_net = resnet34()



        # backbone_net = ViPNAS_ResNet(depth=34)#NAS
        # backbone_net = ViPNAS_Fusion(depth=34)#NAS_Fusion
        # backbone_net = fusion_res(block_type, layers, cfg.network.back_input_channel, cfg.network.back_freeze)
        # backbone_net = ViT(img_size=256,patch_size=32,num_classes=13,)#vision transformer


        #FusionetV2
        # backbone_net = FN_fff(depth=34)
        # backbone_net = cluster_FN(depth=34)
        backbone_net = FN_edge(depth=34)
        # backbone_net = cluster_edge(depth=34)
        # backbone_net = FN_linear(depth=34)
        # backbone_net = fusionNetv2(depth=34)
        # backbone_net = vip_edge(depth=34)
        # backbone_net = Res_edge(block_type, layers, cfg.network.back_input_channel, cfg.network.back_freeze)
        # backbone_net = linear_FN(depth=34)
        # backbone_net = tcformer_light()
        # backbone_net = pvt(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        #             qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])


        #以下的模型在测试阶段
        # backbone_net = TokenPose_B()
        # backbone_net = get_pose_net(cfg)
        # backbone_net = SwinModel.from_pretrained("microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft")

        # summary(backbone_net,input_size=(3,256,256),batch_size=1,device='cpu')
        #print(backbone_net)


        if cfg.network.back_freeze:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, backbone_net.parameters()),
                                   'lr': float(cfg.train.lr_backbone)})
        # rotation head net
        rot_head_net = RotHeadNet(mynet_channels[-1], cfg.network.rot_layers_num, cfg.network.rot_filters_num, cfg.network.rot_conv_kernel_size,
                                  cfg.network.rot_output_conv_kernel_size, cfg.network.rot_output_channels, cfg.network.rot_head_freeze)
        if cfg.network.rot_head_freeze:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                                   'lr': float(cfg.train.lr_rot_head)})
        # translation head net
        trans_head_net = TransHeadNet(mynet_channels[-1], cfg.network.trans_layers_num, cfg.network.trans_filters_num, cfg.network.trans_conv_kernel_size,
                                      cfg.network.trans_output_channels, cfg.network.trans_head_freeze)
        if cfg.network.trans_head_freeze:
            for param in trans_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                                   'lr': float(cfg.train.lr_trans_head)})
        # CDPN (Coordinates-based Disentangled Pose Network)
        model = CDPN(backbone_net, rot_head_net, trans_head_net)
        # get optimizer
        if params_lr_list != []:
            optimizer = torch.optim.RMSprop(params_lr_list, alpha=cfg.train.alpha, eps=float(cfg.train.epsilon),
                                            weight_decay=cfg.train.weightDecay, momentum=cfg.train.momentum)

        else:
            optimizer = None

    # model initialization, 我注释掉的
    if cfg.pytorch.load_model != '':
        logger.info("=> loading model '{}'".format(cfg.pytorch.load_model))
        checkpoint = torch.load(cfg.pytorch.load_model, map_location=lambda storage, loc: storage)
        if type(checkpoint) == type({}):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint.state_dict()

        if 'resnet' in cfg.network.arch:
            model_dict = model.state_dict()
            checkpoint = torch.load(cfg.pytorch.load_model, map_location=lambda storage, loc: storage)
            # filter out unnecessary params
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # update state dict
            model_dict.update(filtered_state_dict)
            # load params to net
            model.load_state_dict(model_dict)
    # else:
    #     if 'resnet' in cfg.network.arch:
    #         logger.info("=> loading official model from model zoo for backbone")
    #         _, _, _, name = resnet_spec[cfg.network.back_layers_num]#原来为cfg.network.back_layers_num
    #         official_resnet = model_zoo.load_url(model_urls[name])
    #         # drop original resnet fc layer, add 'None' in case of no fc layer, that will raise error
    #         official_resnet.pop('fc.weight', None)
    #         official_resnet.pop('fc.bias', None)
    #         model.backbone.load_state_dict(official_resncet)
    #下面的为我家的
    # elif my_init_model != '':
    #     logger.info("=> loading my init model {}".format(my_init_model))
    #     checkpoint = torch.load(my_init_model, map_location=lambda storage, loc: storage)
    #     if type(checkpoint) == type({}):
    #         state_dict = checkpoint['state_dict']
    #     else:
    #         state_dict = checkpoint.state_dict()
    #
    #     model_dict = model.state_dict()
    #     # filter out unnecessary params
    #     filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    #     # update state dict
    #     model_dict.update(filtered_state_dict)
    #     # load params to net
    #     model.load_state_dict(model_dict)
    #     logger.info("successfully loading my init model!!!")
    #断点继续训练
    # if cfg.pytorch.load_model != '':
    #     logger.info("=> loading checkpoint'{}'".format(cfg.pytorch.load_model))
    #     checkpoint = torch.load(cfg.pytorch.load_model, map_location=lambda storage, loc: storage)
    #     # print(next(model.parameters()).device)
    #     # print("所有键值:",checkpoint.keys())
    #     print("epoch:", checkpoint['epoch'])
    #     # optimizer.parameters().cuda()
    #     model.load_state_dict(checkpoint['state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loaded checkpoint '{}'"
    #               .format(cfg.pytorch.load_model))






    return model, optimizer#, swin_backbone


def save_model(path, model, optimizer=None, epoch=None):
    if optimizer is None:
        torch.save({'state_dict': model.state_dict()}, path)
    else:
        torch.save({'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, path)

