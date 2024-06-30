import torch
from torchvision import transforms
import numpy as np
import os
import math


class Config:
    def __init__(self, step, dataSetName, misakaNum, epochs, incr):

        self.MisakaNum = misakaNum
        self.dataset_name = dataSetName
        self.step = step
        self.batchsz = 105
        self.incr = incr
        self.classNum = self.incr*(1+self.step)
        self.reserveNum = 2

        self.preClassNum = self.incr * self.step

        self.trainDir = "/home/tx704/zyx/Dataset/"+self.dataset_name+"/train"
        self.testDir = "/home/tx704/zyx/Dataset/"+self.dataset_name+"/test"
        self.inversionDir = "/home/tx704/zyx/Dataset/"+self.dataset_name+"/inversion"#reserve inversion
        self.allClassNum = len(os.listdir(self.trainDir))

        self.swav_path = "./../../swav.pth"
        self.model_path = "./model/" + self.MisakaNum +'-'+str(self.step) + ".pth"

        self.train_transformer = transforms.Compose([
            # transforms.Resize([32, 32]),
            transforms.Resize([224, 224]),
            transforms.RandomCrop([224, 224], padding=28, pad_if_needed=False, fill=0, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # imagenet
                                 std=[0.229, 0.224, 0.225])
        ])

        self.test_transformer = transforms.Compose([
            # transforms.Resize([32, 32]),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # imagenet
                                 std=[0.229, 0.224, 0.225])
        ])

        self.USE_MULTI_GPU = True

        self.optimizer = "AdamW"
        self.lr = 3e-4
        self.maxMixNum=23

        self.epoch_num = epochs



