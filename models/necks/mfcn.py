import torch
import torch.nn as nn
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

# MFCN: multi-scale feature concat network
__all__ = ["MFCN"]

# logger = logging.getLogger("global_logger")
# log_path = "/home/ubuntu/lkf/uniform-3dad/IUF-master-Depth/experiments/MVTec_3DAD/9_1_only_Depth/test.log"
# logger = create_logger("global_logger", log_path )

class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(MFCN, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [
            in_stride / outstrides[0] for in_stride in instrides
        ]  # for resize
        self.scale_factors_xyz = [
            in_stride / outstrides[0] for in_stride in instrides
        ]  # for resize
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]
        self.upsample_list_xyz = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors_xyz
        ]

    def forward(self, input):
        # print(input)
        # logger.info(input)

        # input--- {'filename': , 'height':, 'width':, 'label':, 'clsname':, 'image': tensor([[[[ 1.5810, 'mask': tensor([[[[,
        #            'clslabel': tensor([[0.,  'features': [tensor([[[[ 1.42,  'strides': [2, 4, 8, 16]}
        features = input["features"]
        features_xyz = input["features_xyz"]
        # print('****features[0]******',features[0].shape)  # torch.Size([6, 24, 112, 112])
        # print('****features[1]******',features[1].shape)  # torch.Size([6, 32, 56, 56])
        # print('****features[2]******',features[2].shape)  # torch.Size([6, 56, 28, 28])
        # print('****features[3]******',features[3].shape)  # torch.Size([6, 160, 14, 14])
        # outstrides  [2, 4, 8, 16]    #outblocks [1, 5, 9, 21]
        assert len(self.inplanes) == len(features) # self.inplanes---{"1": 24, "5": 32, "9": 56, "21": 160} len=4

        feature_list = []
        feature_list_xyz = []
        # print("***self.inplanes***",self.inplanes)     # [24, 32, 56, 160]
        # print("***self.outplanes***",self.outplanes)   # [272]
        # print("***self.instrides***",self.instrides)   # [2, 4, 8, 16]
        # print("***self.outstrides***",self.outstrides) # [16]
        # resize & concatenate

        for i in range(len(features)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(features[i])
            # print("***feature_resize***",feature_resize.shape)
            # ***feature_resize*** torch.Size([6, 24, 14, 14])
            # ***feature_resize*** torch.Size([6, 32, 14, 14])
            # ***feature_resize*** torch.Size([6, 56, 14, 14])
            # ***feature_resize*** torch.Size([6, 160, 14, 14])
            feature_list.append(feature_resize)

        for i in range(len(features_xyz)):
            upsample_xyz = self.upsample_list_xyz[i]
            feature_resize_xyz = upsample_xyz(features[i])
            # print("***feature_resize***",feature_resize.shape)
            # ***feature_resize*** torch.Size([6, 24, 14, 14])
            # ***feature_resize*** torch.Size([6, 32, 14, 14])
            # ***feature_resize*** torch.Size([6, 56, 14, 14])
            # ***feature_resize*** torch.Size([6, 160, 14, 14])
            feature_list_xyz.append(feature_resize_xyz)

        feature_align = torch.cat(feature_list, dim=1) #只有一个tensor
        feature_align_xyz = torch.cat(feature_list_xyz, dim=1) #只有一个tensor
        
        # ***feature_align*** torch.Size([6, 272, 14, 14]) # 272 =24+32+56+160
        # print("***feature_align***",feature_align.shape)
        # print("***feature_align_xyz***",feature_align_xyz.shape) #torch.Size([6, 272, 14, 14])

        return {"feature_align": feature_align, "feature_align_xyz": feature_align_xyz,"outplane": self.get_outplanes()}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides
