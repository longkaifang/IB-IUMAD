import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

from PIL import Image
from .ViT import ViT
from .ViT import ViT_xyz
# from models.vmamba_Fusion_efficross import VSSM_Fusion

import logging
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
# logger = logging.getLogger("global_logger")
# log_path = "/home/ubuntu/lkf/uniform-3dad/IUF-master-Depth/experiments/MVTec_3DAD/9_1_only_Depth/test2.log"
# logger = create_logger("global_logger", log_path )

class UniAD(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        # self.inplanes1_1 = inplanes # self.inplanes1_1[0]---272
        # self.instrides1_1 =instrides # instrides1_1[0]---16
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]  #14*14
        self.feature_jitter = feature_jitter # feature_jitter: {scale: 20.0, prob: 1.0}
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        ) # pos_embed_type: learned, hidden_dim: 256, feature_size:[14, 14], feature_tokens:torch.Size([196, 6, 256])
        self.pos_embed_xyz = build_position_embedding_xyz(
            pos_embed_type, feature_size, hidden_dim
        )
        self.save_recon = save_recon

        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        ) # hidden_dim: 256, feature_size:[14, 14], neighbor_mask: {neighbor_size: [7,7],mask: [False, False, False}
        self.transformer_xyz = Transformer_xyz(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj = nn.Linear(inplanes[0], hidden_dim) #inplanes[0] 16  hidden_dim 256
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])
        
        self.vitclassifiy=ViT(inplanes=3)
        self.vitclassifiy_xyz=ViT_xyz(inplanes=3)

        #fuison 
        self.ScaledDotAttn_f=ScaledDotAttn_f(C=6, L=256)
        # self.mamba_fuison=VSSM_Fusion(inplanes=3)

        initialize_from_cfg(self, initializer)
    
    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return 272

    def get_outstrides(self):
        """
        get strides of the output tensor
        """
        return 1
    
    def add_jitter(self, feature_tokens, scale, prob): #可以理解为对特征添加  噪声
        # feature_tokens torch.Size([196, 6, 272])   feature_jitter: {scale: 20.0, prob: 1.0} 
        if random.uniform(0, 1) <= prob: #根据给定的概率决定是否添加噪声
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda() 
            jitter = jitter * feature_norms * scale  #生成噪声
            feature_tokens = feature_tokens + jitter #将生成的噪声加到原始特征上
        return feature_tokens

    def forward(self, input):
        
        # For Classification
        # print(input.keys())
        # logger.info(input) 
        # input--- {'filename': , 'height':, 'width':, 'label':, 'clsname':, 'image': tensor([[[[ 1.5810, 'mask': tensor([[[[,
        #            'clslabel': tensor([[0.,  'features': [tensor([[[[ 1.42,  'strides': [2, 4, 8, 16], 'feature_align': tensor([[[[,
        #             'outplane': [272]   }

        input.update(self.vitclassifiy(input)) #ViT提取特征，获得 'class_out':, 'outputs_map':
        input.update(self.vitclassifiy_xyz(input))
        #其中 'class_out'：torch.Size([6, 10]),  'outputs_map': 有四个，四个 torch.Size都是([6, 224, 224, 256])

        # 更新的input--- --- {'filename': , 'height':, 'width':, 'label':, 'clsname':, 'image': tensor([[[[ 1.5810, 'mask': tensor([[[[,
        #            'clslabel': tensor([[0.,  'features': [tensor([[[[ 1.42,  'strides': [2, 4, 8, 16], 'feature_align': tensor([[[[,
        #             'outplane': [272]，'class_out': tensor([[ 0.6245, 'outputs_map': [tensor([[[[-4  }
        # logger.info(input) 
        # print(input.keys())
        
        class_condition = input["outputs_map"]  #'outputs_map': 有四个，四个 torch.Size都是([6, 224, 224, 256]) vit输出的特征
        class_condition_xyz = input["outputs_map_xyz"]

        # fusion_vit=[]
        # for i in range(len(class_condition)):
        #     class_condition_fusion=self.mamba_fuison(rearrange(class_condition[i], 'b h w c -> b c h w'),rearrange(class_condition_xyz[i], 'b h w c -> b c h w'))
        #     out = rearrange(class_condition_fusion, 'b c h w -> b h w c')
        #     fusion_vit.append(out)
        # print("class_condition_fusion_0",fusion_vit[0].shape)  #torch.Size([3, 224, 224, 256])
        # print("class_condition_fusion_1",fusion_vit[1].shape) 
        # print("class_condition_fusion_2",fusion_vit[2].shape) 
        # print("class_condition_fusion_3",fusion_vit[3].shape) 

        # print(len(self.feature_size)) # len 为2 14 14 
        # print("self.feature_size",self.feature_size[0],self.feature_size[1])  # 14 14
        
        # For Reconstruction
        feature_align = input["feature_align"]  # B x C X H x W  #只有一个tensor  torch.Size([6, 272, 14, 14])
        # print("feature_align",feature_align.shape)
        feature_align_xyz = input["feature_align_xyz"] 
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"  #feature_align torch.Size([6, 272, 14, 14])
        )  # (H x W) x B x C    torch.Size([196, 6, 272]) 
        feature_tokens_xyz = rearrange(
            feature_align_xyz, "b c h w -> (h w) b c"  #feature_align torch.Size([6, 272, 14, 14])
        )
        # print("***feature_tokens***",feature_tokens.shape)
        if self.training and self.feature_jitter: # feature_jitter: {scale: 20.0, prob: 1.0} 
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob #feature_tokens torch.Size([196, 6, 272])
            )  #对feature_tokens添加噪声
            feature_tokens_xyz = self.add_jitter(
                feature_tokens_xyz, self.feature_jitter.scale, self.feature_jitter.prob #feature_tokens torch.Size([196, 6, 272])
            )
       
        # print("***self.inplanes[0]**",self.inplanes1_1[0]) # self.inplanes 272
        # print("***self.instrides1_1[0]**",self.instrides1_1[0]) # self.instrides1_1[0] 16
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C   torch.Size([196, 6, 256])
        feature_tokens_xyz = self.input_proj(feature_tokens_xyz)
        # print("***feature_tokens***",feature_tokens.shape) # torch.Size([196, 6, 256])
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C 生成位置编码 pos_embed torch.Size([196, 256])
        pos_embed_xyz = self.pos_embed_xyz(feature_tokens_xyz)
        # print("***pos_embed***",pos_embed.shape) # pos_embed torch.Size([196, 256])
        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed, class_condition # feature_tokens是有噪音的特征，pos_embed位置编码， class_condition是vit输出的特征
        )  # (H x W) x B x C
        # print(output_decoder.shape) # torch.Size([8, 196, 6, 256])
        output_decoder_xyz, _xyz = self.transformer_xyz(
            feature_tokens_xyz, pos_embed_xyz, class_condition_xyz # feature_tokens是有噪音的特征，pos_embed位置编码， class_condition是vit输出的特征
        )
        middle_decoder_feature=output_decoder[0:3,...] # 取前三个----感觉output_decoder[0]，[1]值相同（后期修改一下）
        middle_decoder_feature_xyz=output_decoder_xyz[0:3,...] 
       
        
        middle_decoder_feature_rec_0 = rearrange(
            middle_decoder_feature[0], "(h w) b c -> b c (h w)", h=self.feature_size[0]
        ) # torch.Size([6, 256, 196])
        middle_decoder_feature_rec_1 = rearrange(
            middle_decoder_feature[1], "(h w) b c -> b c (h w)", h=self.feature_size[0]
        )   # torch.Size([6, 256, 196])
        middle_decoder_feature_rec_2 = rearrange(
            middle_decoder_feature[2], "(h w) b c -> b c (h w)", h=self.feature_size[0]
        )    # torch.Size([6, 256, 196])
        middle_decoder_feature_rec_0_xyz = rearrange(
            middle_decoder_feature_xyz[0], "(h w) b c -> b c (h w)", h=self.feature_size[0]
        ) # torch.Size([6, 256, 196])
        middle_decoder_feature_rec_1_xyz = rearrange(
            middle_decoder_feature_xyz[1], "(h w) b c -> b c (h w)", h=self.feature_size[0]
        )   # torch.Size([6, 256, 196])
        middle_decoder_feature_rec_2_xyz = rearrange(
            middle_decoder_feature_xyz[2], "(h w) b c -> b c (h w)", h=self.feature_size[0]
        )    # torch.Size([6, 256, 196])        
        # print("**middle_decoder_feature[0]**",middle_decoder_feature_rec_0.size())
        # print("**middle_decoder_feature[1]**",middle_decoder_feature_rec_0.size())
        # print("**middle_decoder_feature[2]**",middle_decoder_feature_rec_0.size())
    
        output_decoder = output_decoder[3] # [196, 6, 256]
        output_decoder_xyz = output_decoder_xyz[3] # [196, 6, 256]
        # print("**output_decoder**",output_decoder.size())
        
        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C  torch.Size([196, 6, 272])
        feature_rec_tokens_xyz = self.output_proj(output_decoder_xyz)
        # print("**feature_rec_tokens**",feature_rec_tokens.size())
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W  torch.Size([6, 272, 14, 14])
        # print("wwwww",feature_rec.shape)
        feature_rec_xyz = rearrange(
            feature_rec_tokens_xyz, "(h w) b c -> b c h w", h=self.feature_size[0]
        )
        # print("wwwww11",feature_rec_xyz.shape)
        # print("**feature_rec**",feature_rec.size())

        if not self.training and self.save_recon: # save_recon: {save_dir: ./results/9_1_only_Depth/rec, save_interval: 100}
                                                # 保存图像
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)
                # print(feature_rec_np.shape)
                # image = Image.fromarray(feature_rec_np)
                # image.save(os.path.join(save_dir, filename_ + ".png"))             

        #### Fusion way1 addtion
        feature_rec_final=feature_rec+feature_rec_xyz
        #### Fusion way2 ScaledDotAttn
        # feature_rec_final=self.ScaledDotAttn_f(feature_rec,feature_rec_xyz)
        #### Fusion way2 mamba
        # feature_rec_final=self.mamba_fuison(feature_rec,feature_rec_xyz)




#预测
        pred = torch.sqrt(
            torch.sum((feature_rec_final - feature_align) ** 2, dim=1, keepdim=True) #重构图像跟原始图像相减
        )  # B x 1 x H x W  torch.Size([6, 1, 14, 14])
        # print("**pred1**",pred.shape)
        pred = self.upsample(pred)  # B x 1 x H x W  torch.Size([6, 1, 224, 224])
        # print("**pred2**",pred.shape)
        return {
            "feature_rec": feature_rec, # torch.Size([6, 272, 224, 224])
            "feature_rec_xyz": feature_rec_xyz,
            "feature_align": feature_align, # torch.Size([6, 272, 14, 14])
            "feature_align_xyz": feature_align_xyz,
            "pred": pred, # torch.Size([6, 1, 224, 224])
            "class_out": input["class_out"], # torch.Size([6, 10])
            "middle_decoder_feature_0": middle_decoder_feature_rec_0, # torch.Size([6, 256, 196])
            "middle_decoder_feature_1": middle_decoder_feature_rec_1, # torch.Size([6, 256, 196])
            "middle_decoder_feature_2": middle_decoder_feature_rec_2, # torch.Size([6, 256, 196])
            "middle_decoder_feature_0_xyz": middle_decoder_feature_rec_0_xyz, # torch.Size([6, 256, 196])
            "middle_decoder_feature_1_xyz": middle_decoder_feature_rec_1_xyz, # torch.Size([6, 256, 196])
            "middle_decoder_feature_2_xyz": middle_decoder_feature_rec_2_xyz, # torch.Size([6, 256, 196])
        }


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim, # 256
        feature_size, # [14, 14]
        neighbor_mask, # {neighbor_size: [7,7],mask: [False, False, False}
        nhead, # 8
        num_encoder_layers, # 4
        num_decoder_layers, # 4
        dim_feedforward, # 1024
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
    ): 
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )# hidden_dim: 256, nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        ) #encoder_layer: TransformerEncoderLayer, num_encoder_layers: 4, encoder_norm: None
        #self.encoder--torch.Size([196, 6, 256])

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )# hidden_dim: 256, feature_size:[14, 14], nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )# decoder_layer: TransformerDecoderLayer, num_decoder_layers: 4, decoder_norm: nn.LayerNorm(hidden_dim), return_intermediate: True

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.pool = nn.AdaptiveAvgPool2d((14, 14))

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        #feature_size:[14, 14] neighbor_size: [7,7]
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, src, pos_embed, class_condition):
#src=feature_tokens---torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 256]), class_condition--4个--torch.Size都是([6, 224, 224, 256])
#src是efficentnet输出的特征,用mfcn级联了四个多尺度生成的异常图像。class_condition的ViT提取的特征
        # 缩小的目标大小为[4, 14, 14, 256]
        target_size = (14, 14)
        output_tensor = []
        
        # class_condition = class_condition.permute(1, 0, 2, 3, 4)

        # print(class_condition[0].size()) #--torch.Size都是([6, 224, 224, 256])
        # print(class_condition[1].size())
        # print(class_condition[2].size())
        # print(class_condition[3].size())
        
        # 使用插值函数进行缩小操作
        for i in range(4):
            result= self.pool(class_condition[i].permute(0, 3, 1, 2))
            result = rearrange(result.permute(0, 2, 3, 1), 'b h w c -> b (h w) c')  # torch.Size([6, 196, 256])
            result = rearrange(result, 'b hw c -> hw b c')  # torch.Size([196, 6, 256]) 
            output_tensor.append(result) #output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256]) 
        # print(output_tensor[0].shape,output_tensor[1].shape,output_tensor[2].shape,output_tensor[3].shape)
                
        
        _, batch_size, _ = src.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1 #pos_embed---torch.Size([196, 256])
        )  # (H X W) x B x C #  pos_embed torch.Size([196, 6, 256])
        # print(pos_embed.shape) # pos_embed.shape----torch.Size([196, 6, 256])

        if self.neighbor_mask: #检查neighbor_mask定义是否存在,存在则继续生成和分配掩码
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )#feature_size:[14, 14] neighbor_size: [7,7]
            # print("**mask**",mask.size()) # torch.Size([196, 196])
            mask_enc = mask if self.neighbor_mask.mask[0] else None #self.neighbor_mask.mask[0]为ture,分配掩码mask
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
            # print("**mask_enc**",mask_enc) # None
            # print("**mask**",mask_dec1) # None
            # print("**mask_dec1**",mask_dec1) # None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder( #src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
            # mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256]) 
            src, mask=mask_enc, pos=pos_embed, output_tensor=output_tensor
        )  # (H X W) x B x C  # torch.Size([196, 6, 256])
        # print("***output_encoder**",output_encoder.size()) # torch.Size([196, 6, 256])
        
        output_decoder = self.decoder(
            output_encoder,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed,
        )  # (H X W) x B x C
        # print("**output_decoder**",output_decoder.shape) # torch.Size([8, 196, 6, 256])
        # print(len(output_decoder)) #8 [output1,norm(output1),output2,norm(output2),output3,norm(output3),output4,norm(output4)]
        # print("**output_decoder_0**",output_decoder[0].shape) # torch.Size([196, 6, 256])
        # print("**output_decoder_1**",output_decoder[1].shape) # torch.Size([196, 6, 256])
        # print("**output_decoder_2**",output_decoder[2].shape) # torch.Size([196, 6, 256])
        # print("**output_decoder_3**",output_decoder[3].shape) # torch.Size([196, 6, 256])
        #output_encoder---torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None

        return output_decoder, output_encoder


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm # None
# encoder_layer: TransformerEncoderLayer, num_encoder_layers: 4, encoder_norm: None
    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        output_tensor: Optional[list] = [],
    ):
        # src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
        # src_key_padding_mask=mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256])
        output = src
        # print(output.size()) #torch.Size([196, 6, 256])
        class_tensor = output_tensor
        # print(class_tensor[0].size(),class_tensor[1].size(),class_tensor[2].size(),class_tensor[3].size()) #torch.Size([196, 6, 256])
        # print(src_key_padding_mask) #None

        for num, layer in enumerate(self.layers): #num=0,1,2,3
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
                class_condition = class_tensor[num]
            ) #循环四次，第一次class_condition[0],output, .....
            # print("**output**",output.size()) # torch.Size([196, 6, 256])

        # print("**output111**",output.size()) # torch.Size([196, 6, 256])
        if self.norm is not None:
            output = self.norm(output)
        # print("**output_final**",output.size()) # torch.Size([196, 6, 256])

        return output #output的最终输出是第四次循环的输出


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
# decoder_layer: TransformerDecoderLayer, num_decoder_layers: 4, decoder_norm: nn.LayerNorm(hidden_dim), return_intermediate: True
    def forward(
        self,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
#output_encoder---torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None
        output = memory

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
# memory--torch.Size([196, 6, 256]),output--torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None
            # print("**output**",output.size()) #torch.Size([196, 6, 256])
            if self.return_intermediate:
                if self.norm is None: #self.norm不为空
                    intermediate.append(output) #不执行
                else:
                    intermediate.append(self.norm(output)) #执行
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # print("**self_norm**",self.norm)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop() #移除intermediate列表中的最后一个元素
                intermediate.append(output)

        
        if self.return_intermediate: 
            # print(len(intermediate)) #8
            # print("**intermediate*",torch.stack(intermediate).shape) # torch.Size([8, 196, 6, 256])
            return torch.stack(intermediate)

        return output #self.return_intermediate为ture返回torch.stack(intermediate)，否则返回output

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ): # hidden_dim: 256, nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        class_condition: Optional[list] = None,
    ):
    # src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
    # src_key_padding_mask=mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256])
        q = k = self.with_pos_embed(src, pos)
        # print("q",q.size())
        # print("k",k.size())
        # print("v",src.size())
        
        q = q * class_condition
        # print("Q",q.size())
        # print("class_condition",class_condition.size())
                
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        
        # print("src_mask",src_mask.size())

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # print("*******src*******",src.size()) # src---torch.Size([196, 6, 256])
        
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        class_condition: Optional[Tensor] = None,
    ):
          
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        
        q = q * class_condition
        # print("Q",q.size())
        # print("class_condition",class_condition.size())
        
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        class_condition: Optional[list] = None,
    ):
        # src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
        # src_key_padding_mask=mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256])
        if self.normalize_before: # normalize_before=False
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, class_condition)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, class_condition)

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
# hidden_dim: 256, feature_size:[14, 14], nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
# memory--torch.Size([196, 6, 256]),output--torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # print("***tgt****",tgt.size()) # tgt---torch.Size([196, 6, 256])
        return tgt

    def forward_pre(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            out,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )

###################### xyz# xyz# xyz# xyz# xyz
# xyz
class Transformer_xyz(nn.Module):
    def __init__(
        self,
        hidden_dim, # 256
        feature_size, # [14, 14]
        neighbor_mask, # {neighbor_size: [7,7],mask: [False, False, False}
        nhead, # 8
        num_encoder_layers, # 4
        num_decoder_layers, # 4
        dim_feedforward, # 1024
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
    ): 
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer_xyz(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )# hidden_dim: 256, nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder_xyz(
            encoder_layer, num_encoder_layers, encoder_norm
        ) #encoder_layer: TransformerEncoderLayer, num_encoder_layers: 4, encoder_norm: None
        #self.encoder--torch.Size([196, 6, 256])

        decoder_layer = TransformerDecoderLayer_xyz(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )# hidden_dim: 256, feature_size:[14, 14], nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder_xyz(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )# decoder_layer: TransformerDecoderLayer, num_decoder_layers: 4, decoder_norm: nn.LayerNorm(hidden_dim), return_intermediate: True

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.pool = nn.AdaptiveAvgPool2d((14, 14))

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        #feature_size:[14, 14] neighbor_size: [7,7]
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, src, pos_embed, class_condition):
#src=feature_tokens---torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 256]), class_condition--4个--torch.Size都是([6, 224, 224, 256])
#src是efficentnet输出的特征,用mfcn级联了四个多尺度生成的异常图像。class_condition的ViT提取的特征
        # 缩小的目标大小为[4, 14, 14, 256]
        target_size = (14, 14)
        output_tensor = []
        
        # class_condition = class_condition.permute(1, 0, 2, 3, 4)

        # print(class_condition[0].size()) #--torch.Size都是([6, 224, 224, 256])
        # print(class_condition[1].size())
        # print(class_condition[2].size())
        # print(class_condition[3].size())
        
        # 使用插值函数进行缩小操作
        for i in range(4):
            result= self.pool(class_condition[i].permute(0, 3, 1, 2))
            result = rearrange(result.permute(0, 2, 3, 1), 'b h w c -> b (h w) c')  # torch.Size([6, 196, 256])
            result = rearrange(result, 'b hw c -> hw b c')  # torch.Size([196, 6, 256]) 
            output_tensor.append(result) #output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256]) 
        # print(output_tensor[0].shape,output_tensor[1].shape,output_tensor[2].shape,output_tensor[3].shape)
                
        
        _, batch_size, _ = src.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1 #pos_embed---torch.Size([196, 256])
        )  # (H X W) x B x C #  pos_embed torch.Size([196, 6, 256])
        # print(pos_embed.shape) # pos_embed.shape----torch.Size([196, 6, 256])

        if self.neighbor_mask: #检查neighbor_mask定义是否存在,存在则继续生成和分配掩码
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )#feature_size:[14, 14] neighbor_size: [7,7]
            # print("**mask**",mask.size()) # torch.Size([196, 196])
            mask_enc = mask if self.neighbor_mask.mask[0] else None #self.neighbor_mask.mask[0]为ture,分配掩码mask
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
            # print("**mask_enc**",mask_enc) # None
            # print("**mask**",mask_dec1) # None
            # print("**mask_dec1**",mask_dec1) # None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder( #src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
            # mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256]) 
            src, mask=mask_enc, pos=pos_embed, output_tensor=output_tensor
        )  # (H X W) x B x C  # torch.Size([196, 6, 256])
        # print("***output_encoder**",output_encoder.size()) # torch.Size([196, 6, 256])
        
        output_decoder = self.decoder(
            output_encoder,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed,
        )  # (H X W) x B x C
        # print("**output_decoder**",output_decoder.shape) # torch.Size([8, 196, 6, 256])
        # print(len(output_decoder)) #8 [output1,norm(output1),output2,norm(output2),output3,norm(output3),output4,norm(output4)]
        # print("**output_decoder_0**",output_decoder[0].shape) # torch.Size([196, 6, 256])
        # print("**output_decoder_1**",output_decoder[1].shape) # torch.Size([196, 6, 256])
        # print("**output_decoder_2**",output_decoder[2].shape) # torch.Size([196, 6, 256])
        # print("**output_decoder_3**",output_decoder[3].shape) # torch.Size([196, 6, 256])
        #output_encoder---torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None

        return output_decoder, output_encoder


class TransformerEncoder_xyz(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm # None
# encoder_layer: TransformerEncoderLayer, num_encoder_layers: 4, encoder_norm: None
    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        output_tensor: Optional[list] = [],
    ):
        # src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
        # src_key_padding_mask=mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256])
        output = src
        # print(output.size()) #torch.Size([196, 6, 256])
        class_tensor = output_tensor
        # print(class_tensor[0].size(),class_tensor[1].size(),class_tensor[2].size(),class_tensor[3].size()) #torch.Size([196, 6, 256])
        # print(src_key_padding_mask) #None

        for num, layer in enumerate(self.layers): #num=0,1,2,3
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
                class_condition = class_tensor[num]
            ) #循环四次，第一次class_condition[0],output, .....
            # print("**output**",output.size()) # torch.Size([196, 6, 256])

        # print("**output111**",output.size()) # torch.Size([196, 6, 256])
        if self.norm is not None:
            output = self.norm(output)
        # print("**output_final**",output.size()) # torch.Size([196, 6, 256])

        return output #output的最终输出是第四次循环的输出


class TransformerDecoder_xyz(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
# decoder_layer: TransformerDecoderLayer, num_decoder_layers: 4, decoder_norm: nn.LayerNorm(hidden_dim), return_intermediate: True
    def forward(
        self,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
#output_encoder---torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None
        output = memory

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
# memory--torch.Size([196, 6, 256]),output--torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None
            # print("**output**",output.size()) #torch.Size([196, 6, 256])
            if self.return_intermediate:
                if self.norm is None: #self.norm不为空
                    intermediate.append(output) #不执行
                else:
                    intermediate.append(self.norm(output)) #执行
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # print("**self_norm**",self.norm)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop() #移除intermediate列表中的最后一个元素
                intermediate.append(output)

        
        if self.return_intermediate: 
            # print(len(intermediate)) #8
            # print("**intermediate*",torch.stack(intermediate).shape) # torch.Size([8, 196, 6, 256])
            return torch.stack(intermediate)

        return output #self.return_intermediate为ture返回torch.stack(intermediate)，否则返回output

class TransformerEncoderLayer_xyz(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ): # hidden_dim: 256, nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        class_condition: Optional[list] = None,
    ):
    # src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
    # src_key_padding_mask=mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256])
        q = k = self.with_pos_embed(src, pos)
        # print("q",q.size())
        # print("k",k.size())
        # print("v",src.size())
        
        q = q * class_condition
        # print("Q",q.size())
        # print("class_condition",class_condition.size())
                
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        
        # print("src_mask",src_mask.size())

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # print("*******src*******",src.size()) # src---torch.Size([196, 6, 256])
        
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        class_condition: Optional[Tensor] = None,
    ):
          
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        
        q = q * class_condition
        # print("Q",q.size())
        # print("class_condition",class_condition.size())
        
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        class_condition: Optional[list] = None,
    ):
        # src=feature_tokens---torch.Size([196, 6, 256]),  pos_embed.shape----torch.Size([196, 6, 256])
        # src_key_padding_mask=mask=mask_enc=None  output_tensor[0/1/2/3]的size都是torch.Size([196, 6, 256])
        if self.normalize_before: # normalize_before=False
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, class_condition)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, class_condition)

class TransformerDecoderLayer_xyz(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
# hidden_dim: 256, feature_size:[14, 14], nhead: 8, dim_feedforward: 1024, dropout: 0.1, activation: relu, normalize_before: False
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
# memory--torch.Size([196, 6, 256]),output--torch.Size([196, 6, 256]), pos_embed---torch.Size([196, 6, 256]), tgt_mask=emory_mask=None
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # print("***tgt****",tgt.size()) # tgt---torch.Size([196, 6, 256])
        return tgt

    def forward_pre(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            out,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


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
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        # pos_embed_type: learned, hidden_dim: 256, feature_size:[14, 14], feature_tokens:torch.Size([196, 6, 256])
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos


def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    # pos_embed_type: learned, hidden_dim: 256, feature_size:[14, 14], feature_tokens:torch.Size([196, 6, 256])
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"): # 可学习位置编码
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed

####  xyz
class PositionEmbeddingSine_xyz(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class PositionEmbeddingLearned_xyz(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        # pos_embed_type: learned, hidden_dim: 256, feature_size:[14, 14], feature_tokens:torch.Size([196, 6, 256])
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos

def build_position_embedding_xyz(pos_embed_type, feature_size, hidden_dim):
    # pos_embed_type: learned, hidden_dim: 256, feature_size:[14, 14], feature_tokens:torch.Size([196, 6, 256])
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine_xyz(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"): # 可学习位置编码
        pos_embed = PositionEmbeddingLearned_xyz(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed



#### Fusion module
class ScaledDotAttn_f(nn.Module):

    def __init__(self, C, L):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        # self.ln = nn.LayerNorm([C, L])

    def forward(self, x, y):
        # trans pose C to last dim
        # q = x.transpose(1, 2)
        q = x
        k = y
        v = y
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k) / math.sqrt(d_k)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # out = out.transpose(1, 2)
        out = self.dropout(out)
        # out = self.ln(out)

        return out 