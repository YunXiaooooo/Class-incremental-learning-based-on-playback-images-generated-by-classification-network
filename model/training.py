import torch
from torch import nn, optim
from model.modelseting import *
from tqdm import tqdm
import numpy as np
from data.DataSet import getDataFromFileAboutClass
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import os, shutil
import math



def finetuning(config, teacher, student, train_data, test_data, log):
    log.printInfo("teacher")
    log.printInfo(teacher)
    log.printInfo("student")
    log.printInfo(student)
    teacher.eval()

    optimizer = optim.AdamW(student.parameters(), lr=config.lr)

    # swav = load_swav(config)
    # if config.USE_MULTI_GPU:
    #     swav = torch.nn.DataParallel(swav).cuda()  # 多卡
    # else:
    #     swav = swav.cuda()  # 单卡
    # swav.eval()

    end_epoch = config.epoch_num
    ratio = 3
    bestAcc = 0
    for epoch in tqdm(range(0, end_epoch), ncols=50, leave=True):
        student.train()

        total_loss1 = torch.tensor([0]).cuda()
        total_loss2 = torch.tensor([0]).cuda()
        total_loss3 = torch.tensor([0]).cuda()
        total_sampleNum = 0
        total_correct = 0
        for batchidx, (x, label) in enumerate(train_data):
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            x, label = x.cuda(), label.cuda()

            if len(label)<config.batchsz//3 and epoch==end_epoch-1:
                log.printInfo("drop last batc in last epoch")
                break

            oldIdx = torch.where(label < config.preClassNum)[0]
            newIdx = torch.where(label >= config.preClassNum)[0]
            with torch.no_grad():
                x, distlliationIdx, label = mixup(x, label, oldIdx, newIdx, config.maxMixNum)


            student_out, student_xout = student(x)
            with torch.no_grad():
                teacher_out, teacher_xout = teacher(x)

            loss_old = cosineLossFunc(student_out[oldIdx], label[oldIdx], config.classNum)
            loss_new = cosineLossFunc(student_out[newIdx], label[newIdx], config.classNum)
            loss1 = 0.8 * loss_old+loss_new
            loss2 = distillationLossFunc(teacher_xout[distlliationIdx], student_xout[distlliationIdx]) * ratio

            loss3 = cosineDistillationLossFunc(teacher_out[distlliationIdx], student_out[distlliationIdx], config.preClassNum)


            loss = loss1+loss2


            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # loss 累加
                total_loss1 = total_loss1 + loss1.detach()
                total_loss2 = total_loss2 + loss2.detach()
                total_loss3 = total_loss3 + loss3.detach()
                pred = student_out.argmax(dim=1)
                correct = torch.eq(pred[oldIdx], label[oldIdx]).float().sum().item() + \
                          torch.eq(pred[newIdx], label[newIdx]).float().sum().item()
                total_correct += correct
                total_sampleNum += (len(oldIdx) + len(newIdx))

        with torch.no_grad():
            train_acc = total_correct/total_sampleNum
            str = '%d \n:loss1=%.5f, loss2=%.5f, loss3=%.5f, trainAcc=%.5f' %(epoch, total_loss1.item(), total_loss2.item(), total_loss3.item(),  train_acc)

        student.eval()
        teacher.eval()
        with torch.no_grad():
            # test
            total_loss1 = torch.tensor([0]).cuda()
            total_loss2 = torch.tensor([0]).cuda()
            total_loss3 = torch.tensor([0]).cuda()
            total_sampleNum = 0
            total_correct = 0
            for x, label in test_data:
                x, label = x.cuda(), label.cuda()
                oldIdx = torch.where(label < config.preClassNum)[0]
                newIdx = torch.where(label >= config.preClassNum)[0]

                # x, distlliationIdx = mixup(x, label, oldIdx, newIdx)

                student_out, student_xout = student(x)
                teacher_out, teacher_xout = teacher(x)

                loss_old = cosineLossFunc(student_out[oldIdx], label[oldIdx], config.classNum)
                loss_new = cosineLossFunc(student_out[newIdx], label[newIdx], config.classNum)
                loss1 = 0.8 * loss_old + loss_new
                loss2 = distillationLossFunc(teacher_xout[oldIdx], student_xout[oldIdx]) * ratio
                loss3 = cosineDistillationLossFunc(teacher_out[oldIdx], student_out[oldIdx], config.preClassNum)

                loss = loss1 + loss2 + loss3

                # loss 累加
                total_loss1 = total_loss1 + loss1.detach()
                total_loss2 = total_loss2 + loss2.detach()
                total_loss3 = total_loss3 + loss3.detach()
                pred = student_out.argmax(dim=1)
                correct = torch.eq(pred[oldIdx], label[oldIdx]).float().sum().item()+torch.eq(pred[newIdx], label[newIdx]).float().sum().item()
                total_correct += correct
                total_sampleNum += (len(oldIdx)+len(newIdx))

            test_acc = total_correct/total_sampleNum
            str += ', loss1=%.5f,loss2=%.5f,,loss3=%.5fvalidAcc=%.5f' % (total_loss1.item(), total_loss2.item(), total_loss3.item(), test_acc)

            log.printInfo(str)
            if(bestAcc<=test_acc and epoch>end_epoch-20):
                model_save(student, config.model_path, config.USE_MULTI_GPU)
                bestAcc = test_acc





def cosineLossFunc(cosine, label, classNum):
    if not torch.any(label):
        # print("cosineLossFunc input none")
        return torch.Tensor([0]).cuda()

    label_onehot = nn.functional.one_hot(label, num_classes=classNum)

    loss_positive = torch.sum(torch.mul(cosine, label_onehot), dim=1, keepdim=True)
    loss_positive = torch.pow(1-loss_positive, 3)
    loss_positive = torch.sum(loss_positive) / cosine.shape[0]

    loss_negative = torch.mul(cosine, 1-label_onehot)
    loss_negative = torch.sum(torch.pow(torch.abs(loss_negative), 2), dim=1, keepdim=True)/(classNum-1)
    loss_negative = torch.sum(loss_negative) / cosine.shape[0]

    loss = loss_positive+2*loss_negative

    return loss


def distillationLossFunc(teacherXout, studentXout):
    if not torch.any(teacherXout):
        # print("distillationLossFunc input none")
        return torch.Tensor([0]).cuda()
    # loss = nn.MSELoss()(studentXout, teacherXout)
    teacherNorm = torch.linalg.norm(teacherXout, dim=1, keepdim=True)
    teacherNorm = teacherNorm.clamp(min=1e-12)
    TXout = torch.div(teacherXout, teacherNorm)

    studentNorm = torch.linalg.norm(studentXout, dim=1, keepdim=True)
    studentNorm = studentNorm.clamp(min=1e-12)
    SXout = torch.div(studentXout, studentNorm)

    loss = 1-torch.sum(torch.mul(TXout,SXout), dim=1, keepdim=True)
    # loss = torch.pow(loss, 3)
    loss = torch.mean(loss)

    return loss

def cosineDistillationLossFunc(teacher_out, student_out, preClassNum):
    if not torch.any(teacher_out):
        # print("distillationLossFunc input none")
        return torch.Tensor([0]).cuda()

    student_out = student_out[:,:preClassNum]
    loss = nn.MSELoss()(teacher_out, student_out)
    # loss = torch.pow(teacherXout-studentXout, 2)
    # loss = torch.sum(loss)/loss.shape[0]
    return loss


def mixup(x, label, oldIdx, newIdx, maxMixNum):
    if not torch.any(oldIdx):
        # print("mixup input none")
        return x, oldIdx, label


    mixLabel = copy.deepcopy(label[oldIdx[:maxMixNum]])
    mixX = copy.deepcopy((x[oldIdx[:maxMixNum]]))
    if len(oldIdx)<maxMixNum:
        repeatNum = maxMixNum // mixX.shape[0]
        mixX = mixX.repeat(repeatNum+1, 1, 1, 1)
        mixLabel = mixLabel.repeat(repeatNum+1)
    mixX = mixX[:maxMixNum]
    mixLabel = mixLabel[:maxMixNum]

   # tmpX = copy.deepcopy(x)
   # if tmpX.shape[0]<maxMixNum:
   #     repeatNum = maxMixNum // tmpX.shape[0]
   #     tmpX = tmpX.repeat(repeatNum + 1, 1, 1, 1)
   #
   # tmpX = tmpX[:maxMixNum]


    mixIdx = torch.arange(0, maxMixNum).cuda()
    randIdx = torch.randperm(maxMixNum)


    # lam = torch.rand(mixNum, 1).cuda()
    lam = torch.Tensor(np.random.beta(1,1,[maxMixNum, 1])).cuda()
    lam = lam.unsqueeze(-1).unsqueeze(-1)
    # lam = torch.Tensor(np.random.beta(1, 1, mixX.shape)).cuda()

    mixX = mixX*lam+mixX[randIdx]*(1-lam)

    distlliationIdx = torch.cat((oldIdx, mixIdx + len(label)))
    x = torch.cat((x,mixX),dim=0)
    label = torch.cat((label, mixLabel), dim=0)

    return x, distlliationIdx, label


def loss3Fun(resnet1, resnet2):
    ratio = 0.01
    loss = torch.Tensor([0]).cuda()
    m1 = resnet1.conv1
    m2 = resnet2.conv1
    for param1, param2 in zip(m1.parameters(), m2.parameters()):
        diff = param1 - param2
        loss = loss + torch.sum(diff ** 2) * ratio * 0.5

    layer_1 = resnet1.layer1
    layer_2 = resnet2.layer1
    for m1, m2 in zip(layer_1.children(), layer_2.children()):
        if isinstance(m1, torch.nn.BatchNorm2d):
            continue
        for param1, param2 in zip(m1.parameters(), m2.parameters()):
            diff = param1 - param2
            loss = loss + torch.sum(diff ** 2) * ratio * 0.5

    layer_1 = resnet1.layer2
    layer_2 = resnet2.layer2
    for m1, m2 in zip(layer_1.children(), layer_2.children()):
        if isinstance(m1, torch.nn.BatchNorm2d):
            continue
        for param1, param2 in zip(m1.parameters(), m2.parameters()):
            diff = param1 - param2
            loss = loss + torch.sum(diff ** 2) * ratio * 0.5

    layer_1 = resnet1.layer3
    layer_2 = resnet2.layer3
    for m1, m2 in zip(layer_1.children(), layer_2.children()):
        if isinstance(m1, torch.nn.BatchNorm2d):
            continue
        for param1, param2 in zip(m1.parameters(), m2.parameters()):
            diff = param1 - param2
            loss = loss + torch.sum(diff ** 2) * ratio * 0.2

    layer_1 = resnet1.layer4
    layer_2 = resnet2.layer4
    for m1, m2 in zip(layer_1.children(), layer_2.children()):
        if isinstance(m1, torch.nn.BatchNorm2d):
            continue
        for param1, param2 in zip(m1.parameters(), m2.parameters()):
            diff = param1 - param2
            loss = loss + torch.sum(diff ** 2) * ratio * 0.1




    return loss



# def mixup(x, label, oldIdx, newIdx, maxMixNum):
#     if not torch.any(oldIdx):
#         # print("mixup input none")
#         return x, oldIdx, label
#
#     mixNum = min(len(oldIdx), maxMixNum)
#
#     mixIdx = torch.arange(0, mixNum).cuda()
#     mixLabel = copy.deepcopy(label[oldIdx[:mixNum]])
#     mixX = copy.deepcopy((x[oldIdx[:mixNum]]))
#
#     randIdx = torch.randperm(mixNum)
#     # lam = torch.rand(mixNum, 1).cuda()
#     lam = torch.Tensor(np.random.beta(1,1,[mixNum, 1])).cuda()
#     lam = lam.unsqueeze(-1).unsqueeze(-1)
#     # lam = torch.Tensor(np.random.beta(1, 1, mixX.shape)).cuda()
#
#     mixX = mixX*lam+mixX[randIdx]*(1-lam)
#
#     distlliationIdx = torch.cat((oldIdx, mixIdx + len(label)))
#     x = torch.cat((x,mixX),dim=0)
#     label = torch.cat((label, mixLabel), dim=0)
#
#     return x, distlliationIdx, label


def stepTest(config, log):
    model_path = "./model/" + config.MisakaNum + '-' + str(config.step) + ".pth"
    model = model_load(model_path)
    if config.USE_MULTI_GPU:
        model = torch.nn.DataParallel(model).cuda()  # 多卡
    else:
        model = model.cuda()  # 单卡

    for i in range(0, config.step+1):
        LS = i*config.incr
        LE = (i+1)*config.incr
        if(config.allClassNum-LE<config.incr):
            LE = config.allClassNum
        test_dataloader = getDataFromFileAboutClass(config.testDir, config.test_transformer,
                                                    LS, LE, config.batchsz)
        model.eval()
        with torch.no_grad():
            total_sampleNum = 0
            total_correct = 0
            for x, label, _ in test_dataloader:
                x, label = x.cuda(), label.cuda()

                out = model(x)

                pred = out.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_sampleNum += x.shape[0]

            test_acc = total_correct / total_sampleNum
        logstr = 'step=%d, testAcc=%.5f' % (i, test_acc)
        # print(logstr)
        log.printInfo(logstr)

    test_dataloader = getDataFromFileAboutClass(config.testDir, config.test_transformer,
                                                0, config.classNum, config.batchsz)
    model.eval()
    with torch.no_grad():
        total_sampleNum = 0
        total_correct = 0
        for x, label, filePath in test_dataloader:
            x, label = x.cuda(), label.cuda()

            out = model(x)

            pred = out.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_sampleNum += x.shape[0]

        test_acc = total_correct / total_sampleNum
    logstr = 'all_testAcc=%.5f' % (test_acc)
    # print(logstr)
    log.printInfo(logstr)

def select_img(config):
    model_path = "./model/" + config.MisakaNum + '-' + str(config.step) + ".pth"
    model = model_load(model_path)
    model = MisakaNet_Teacher(model)

    if config.USE_MULTI_GPU:
        model = torch.nn.DataParallel(model).cuda()  # 多卡
    else:
        model = model.cuda()  # 单卡

    LE = config.classNum
    LS = config.incr*config.step
    for c in range(LS, LE):
        train_dataloader = getDataFromFileAboutClass(config.trainDir, config.test_transformer,
                                                    c, c+1, config.batchsz)
        model.eval()
        with torch.no_grad():
            total_sampleNum = 0
            featureMean = torch.zeros([1, 2048]).cuda()
            features = torch.Tensor([]).cuda()
            path = []
            for x, label, filePath in train_dataloader:
                x, label = x.cuda(), label.cuda()

                out, xout = model(x)

                featureMean = featureMean+torch.sum(xout, dim=0)
                total_sampleNum += x.shape[0]

                features = torch.cat((features, xout), dim=0)
                filePath = list(filePath)
                path = path+filePath

            featureMean = featureMean / total_sampleNum


            normFeatures = torch.div(features, torch.norm(features, p=2, dim=1, keepdim=True))
            normFeatureMean = torch.div(featureMean, torch.norm(featureMean, p=2, dim=1, keepdim=True))
            dist = 1-torch.mm(normFeatures, normFeatureMean.t())

            # dist = torch.norm(features-featureMean, p=2, dim=1, keepdim=True)
            dist = dist.t()
            dist = dist.squeeze(0)
            dist = dist.cpu().numpy()
            top_idx = np.argpartition(dist, -config.reserveNum)[:config.reserveNum]
            save_imgs = [path[i] for i in top_idx]
            print(save_imgs)
            for src in save_imgs:
                dst = src.replace('train', 'reserve2')
                dir = os.path.dirname(src)
                dir = dir.replace('train', 'reserve2')
                if not os.path.exists(dir):
                    os.makedirs(dir)
                shutil.copy(src, dst)


def copyParameter(student, teacther, ratio=0.5):
    for p_student, p_teacher in zip(student.parameters(), teacther.parameters()):
        avg_p = p_student*(1-ratio)+p_teacher*ratio
        p_student.data.copy_(avg_p)
