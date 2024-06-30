from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import math

def getDataFromFile(dir, thisTransformer, config):
    dataset = ImageFolder(dir, transform=thisTransformer)  # data_dir精确到分类目录的上一级
    thisDataloader = DataLoader(dataset, batch_size=config.batchsz, shuffle=True)
    return thisDataloader


class MyDataset(Dataset):
    def __init__(self, path1, path2, transform, config, isEqual=False):
        self.filenames = list()
        self.labels = list()
        self.transform = transform

        preClassFiles = next(os.walk(path1))[1]
        preClassFiles.sort()
        currentClassFiles = next(os.walk(path2))[1]
        currentClassFiles.sort()

        preClassNum = len(preClassFiles)
        config.preClassNum = min(preClassNum, config.preClassNum)
        preClassNum = config.preClassNum
        currentclassNum = len(currentClassFiles)
        config.classNum = min(config.classNum, currentclassNum)
        if len(currentClassFiles) - config.classNum<config.incr:
            config.classNum = len(currentClassFiles)

        classNum = config.classNum
        mSampleNumEachClass = 0
        for i in range(preClassNum, classNum):
            classpath = os.path.join(path2, currentClassFiles[i])
            images = os.listdir(classpath)
            self.filenames = self.filenames+[os.path.join(classpath, img) for img in images ]
            labels = [i for idx in range(len(images))]
            self.labels = self.labels+labels
            mSampleNumEachClass += len(images)

        mSampleNumEachClass = mSampleNumEachClass/(classNum-preClassNum)
        for i in range(0, preClassNum):
            classpath = os.path.join(path1, preClassFiles[i])
            images = os.listdir(classpath)

            if isEqual:
                repeatNum = math.ceil(mSampleNumEachClass / len(images))
                images = images*repeatNum
                # idx = [random.randint(0, len(images)-1) for _ in range(mSampleNumEachClass-len(images))]
                # randomImages = [images[i] for i in randomIdx]
                # images = images+randomImages

            self.filenames = self.filenames + [os.path.join(classpath, img) for img in images]
            labels = [i for idx in range(len(images))]
            self.labels = self.labels+labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

def getTopClassDataFromFile(predir ,currentdir , thisTransformer, config, isEqual=False):
    thisDataloader = DataLoader(MyDataset(predir, currentdir, thisTransformer, config, isEqual),
                                batch_size=config.batchsz, shuffle=True, num_workers=4, prefetch_factor=1)
    return thisDataloader


class StepDataset(Dataset):
    def __init__(self, path, transform, LS, LE):
        self.filenames = list()
        self.labels = list()
        self.transform = transform

        ClassFiles = next(os.walk(path))[1]
        ClassFiles.sort()

        for i in range(LS, LE):
            classpath = os.path.join(path, ClassFiles[i])
            images = os.listdir(classpath)
            self.filenames = self.filenames+[os.path.join(classpath, img) for img in images ]
            labels = [i for idx in range(len(images))]
            self.labels = self.labels+labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx], self.filenames[idx]

def getDataFromFileAboutClass(dir, thisTransformer, LS, LE, bz):
    thisDataloader = DataLoader(StepDataset(dir, thisTransformer, LS, LE),
                                batch_size=bz, shuffle=True, num_workers=4, prefetch_factor=1)
    return thisDataloader