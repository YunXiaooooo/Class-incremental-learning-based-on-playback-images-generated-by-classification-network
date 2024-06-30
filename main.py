from config import Config
from sys import path

path.append('./data')
path.append('./model')
path.append('./tool')

from data.DataSet import getTopClassDataFromFile, getDataFromFileAboutClass
from model.training import *
from log.LogCtrl import *
import os
import argparse
import copy



def main(args):

    cfg = Config(args.step, args.datasetName, args.misakaNum, args.epochs, args.incr)
    log = Misakalog(cfg)

    train_dataloader = getTopClassDataFromFile(cfg.inversionDir, cfg.trainDir, cfg.train_transformer, cfg,
                                               isEqual=False)
    test_dataloader = getTopClassDataFromFile(cfg.testDir, cfg.testDir, cfg.test_transformer, cfg)

    if cfg.step == 0:
        teacher = MisakaNet(cfg)
        student = MisakaNet(cfg)
    else:
        if cfg.step == 1:
            teacher_model_path = "./model/10032-0.pth"

        else:
            teacher_model_path = "./model/" + cfg.MisakaNum + '-' + str(cfg.step - 1) + ".pth"

        teacher = model_load(teacher_model_path)
        preweight = teacher.fc.weight
        student = MisakaNet(cfg, preweight)
        teacher = MisakaNet_Teacher(teacher)

        student = student.cpu()
        teacher = teacher.cpu()
        copyParameter(student.backbone.layer3, teacher.backbone.layer3, ratio=0.5)
        copyParameter(student.backbone.layer4, teacher.backbone.layer4, ratio=0.8)

        # student.backbone = copy.deepcopy(teacher.backbone)

    if cfg.USE_MULTI_GPU:
        teacher = torch.nn.DataParallel(teacher).cuda()  # 多卡
        student = torch.nn.DataParallel(student).cuda()  # 多卡
    else:
        teacher = teacher.cuda()  # 单卡
        student = student.cuda()  # 单卡

    finetuning(cfg, teacher, student, train_dataloader, test_dataloader, log)
    stepTest(cfg, log)
    select_img(cfg)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'  # 如果是多卡改成类似0,1,2
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int)
    parser.add_argument('--misakaNum', type=str, default="10032")
    parser.add_argument('--datasetName', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--incr', type=int)
    args = parser.parse_args()

    if args.step<0:
        cfg = Config(0, args.datasetName, '10032', args.epochs, args.incr)
        select_img(cfg)
    else:
        main(args)




