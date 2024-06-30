import torch
import torchvision
import os
from torch import nn
import numpy as np
import copy
import pickle



def load_swav(config):
    if os.path.exists(config.swav_path):
        model = torch.load(config.swav_path)  # 对应torch.save(model, mymodel.pth)， 保存有模型结构
    else:
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        torch.save(model, config.swav_path)

    model.fc = nn.Identity()
    return model

def convert_relu_to_activition(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, torch.nn.LeakyReLU(0.1))
        else:
            convert_relu_to_activition(child)

def convert_convpad(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.padding_mode = 'replicate'


def convert_bn_momomentum(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.momentum = 0.01


class incrementalDiscriminatorHead(torch.nn.Module):
    def __init__(self, config, preWeight=None):
        super(incrementalDiscriminatorHead, self).__init__()

        fc = nn.Linear(2048, config.classNum - config.preClassNum, bias=False)
        w = fc.weight.data.t()
        self.currentWeight = nn.Parameter(w, requires_grad=True)

        if preWeight is None:
            self.preWeight = None
        else:
            self.preWeight = nn.Parameter(preWeight, requires_grad=False)

    def forward(self, x):
        out = x
        outNorm = torch.linalg.norm(out, dim=1, keepdim=True)
        outNorm = outNorm.clamp(min=1e-12)
        out = torch.div(out, outNorm)

        weightNorm = torch.linalg.norm(self.currentWeight, dim=0, keepdim=True)
        weightNorm = weightNorm.clamp(min=1e-12)
        weight = torch.div(self.currentWeight, weightNorm)
        currentOut = torch.mm(out, weight)
        if self.preWeight is None:
            out = currentOut
        else:
            weightNorm = torch.linalg.norm(self.preWeight, dim=0, keepdim=True)
            weightNorm = weightNorm.clamp(min=1e-12)
            weight = torch.div(self.preWeight, weightNorm)
            preOut = torch.mm(out, weight)

            out = torch.cat((preOut, currentOut), 1)

        return out

class MisakaNet(torch.nn.Module):
    def __init__(self, config, preWeight=None):
        super(MisakaNet, self).__init__()
        self.backbone = load_swav(config)
        # self.backbone = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        self.backbone.fc = nn.Identity()
        self.fc = incrementalDiscriminatorHead(config, preWeight)

        convert_relu_to_activition(self.backbone)


    def forward(self, x):
        xout = self.backbone(x)
        out = self.fc(xout)

        return out, xout

def model_load(model_path):
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print('load success! ')
    else:
        print('No model is saved in \'{}\''.format(model_path))
        model = None
    return model

def model_save(model, model_path, isDataParall):
    if isDataParall:
        model = model.module

    save_model = copy.deepcopy(model.backbone)

    if model.fc.preWeight is None:
        weight = model.fc.currentWeight.data
    else:
        weight = torch.cat((model.fc.preWeight.data, model.fc.currentWeight.data), 1)
    save_model.fc = CosineLinear(weight)

    torch.save(save_model, model_path)
    print('model is saved')




class CosineLinear(nn.Module):
    def __init__(self, weight):
        super(CosineLinear, self).__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)  # Parameter将张量变为可训练的参数 ，tensor为in_features*out_features

    def forward(self, x):
        out = x
        # out = x-self.bias
        outNorm = torch.linalg.norm(out, dim=1, keepdim=True)
        outNorm = outNorm.clamp(min=1e-12)
        out = torch.div(out, outNorm)

        weightNorm = torch.linalg.norm(self.weight, dim=0, keepdim=True)
        weightNorm = weightNorm.clamp(min=1e-12)
        weight = torch.div(self.weight, weightNorm)
        out = torch.mm(out, weight)

        return out


class MisakaNet_Teacher(nn.Module):
    def __init__(self, model):
        super(MisakaNet_Teacher, self).__init__()

        self.backbone = copy.deepcopy(model)
        self.backbone.fc = nn.Identity()
        self.fc = copy.deepcopy(model.fc)

    def forward(self, x):
        xout = self.backbone(x)
        out = self.fc(xout)

        return out, xout


