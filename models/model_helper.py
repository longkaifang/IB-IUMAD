import copy
import importlib

import torch
import torch.nn as nn
from utils.misc_helper import to_device


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]  # backbone neck reconstruction
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname) # backbone frozen------models.backbones.efficientnet_b4
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"]) #如果prev 不为None，则表示当前模块依赖于前一个模块的输出  neck依赖backbone，reconstruction依赖neck
                kwargs["inplanes"] = prev_module.get_outplanes() #将前置模块的输出通道数作为当前模块的输入通道数 
                kwargs["instrides"] = prev_module.get_outstrides() #将前置模块的输出步幅作为当前模块的输入步幅。
                # bockbone的inplanes和instrides在 update_config更新了

            module = self.build(mtype, kwargs) # 相当于执行efficientnet_b4(pretrained=True, outlayers=[1,2,3,4])
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name) # 从加载的模块中获取指定的类。 如果 cls_name 是 efficientnet_b4 则相当于获取 models.backbones.efficientnet_b4 类
        return cls(**kwargs) #

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        input = copy.copy(input)
        # print("**input image***",input["image"].shape)  #torch.Size([6, 3, 224, 224])
        
        if input["image"].device != self.device:
            input = to_device(input, device=self.device)
        for submodule in self.children():
            # print("**input image11***",input["image"].shape)
            output = submodule(input)
            input.update(output)
        return output

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
