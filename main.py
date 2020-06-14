import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import models
import random
from lib.LinearAverage import LinearAverage
from lib.BatchAverage import BatchCriterion
from lib.BatchAverageRot import BatchCriterionRot
from lib.BatchAverageFour import BatchCriterionFour
from lib.utils import AverageMeter
from test import kNN
import numpy as np

from lib.utils import save_checkpoint, adjust_learning_rate
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3201, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float, 
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')

parser.add_argument('--result', default="", type=str)
parser.add_argument('--seedstart', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)


parser.add_argument("--synthesis", action="store_true")
parser.add_argument('--showfeature', action="store_true")
parser.add_argument('--multiaug', action="store_true")
parser.add_argument('--multitask', action="store_true")
parser.add_argument("--multitaskposrot", action="store_true")
best_prec1 = 0

parser.add_argument("--domain", action="store_true")
parser.add_argument("--circle", action="store_true")
parser.add_argument("--saveembed", type=str, default="")


def get_learnable_para(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def main():

    global args, best_prec1
    args = parser.parse_args()

    #  init seed
    my_whole_seed = 222
    random.seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    torch.manual_seed(my_whole_seed)
    torch.cuda.manual_seed_all(my_whole_seed)
    torch.cuda.manual_seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(my_whole_seed)


    for kk_time in range(args.seedstart, args.seedstart+1):
        args.seed = kk_time
        args.result = args.result + str(args.seed)

        # create model
        model = models.__dict__[args.arch](low_dim=args.low_dim, multitask=args.multitask , showfeature=args.showfeature, domain=args.domain,args=args)
        model = torch.nn.DataParallel(model).cuda()
        print ('Number of learnable params', get_learnable_para(model)/1000000., " M")

        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        aug = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize])
        # aug = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.), ratio=(3 / 4, 4 / 3)),
        #                           transforms.RandomHorizontalFlip(p=0.5),
        #                           get_color_distortion(s=1),
        #                           transforms.Lambda(lambda x: gaussian_blur(x)),
        #                           transforms.ToTensor(),
        #                           normalize])
        aug_test = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize])

        # load dataset
        import datasets.fundus_amd_syn_crossvalidation as medicaldata
        train_dataset = medicaldata.traindataset(root=args.data, transform=aug, train=True, args=args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=True if args.multiaug else False,  worker_init_fn=random.seed(my_whole_seed))

        valid_dataset = medicaldata.traindataset(root=args.data, transform=aug_test, train=False, args=args)
        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4,
            worker_init_fn=random.seed(my_whole_seed))


        # define lemniscate and loss function (criterion)
        ndata = train_dataset.__len__()

        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m).cuda()


        if args.synthesis:
            print ("running synthesis")
            criterion = BatchCriterionFour(1, 0.1, args.batch_size, args).cuda()
        elif args.multiaug:
            print ("running cvpr")
            criterion = BatchCriterion(1, 0.1, args.batch_size, args).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                lemniscate = checkpoint['lemniscate']
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if args.evaluate:
            knn_num = 100
            auc, acc, precision, recall, f1score = kNN(args, model, lemniscate, train_loader, val_loader, knn_num, args.nce_t, 2)
            print ("auc", auc)
            f = open("savedmodels/result.txt", "a+")
            f.write("auc: %.4f\n" % (auc))
            f.write("acc: %.4f\n" % (acc))
            f.write("pre: %.4f\n" % (precision))
            f.write("recall: %.4f\n" % (recall))
            f.write("f1score: %.4f\n" % (f1score))
            f.close()
            return

        # mkdir result folder and tensorboard
        os.makedirs(args.result, exist_ok=True)
        writer = SummaryWriter("runs/" + str(args.result.split("/")[-1]))
        writer.add_text('Text', str(args))

        # copy code
        import shutil, glob
        source = glob.glob("*.py")
        source += glob.glob("*/*.py")
        os.makedirs(args.result + "/code_file", exist_ok=True)
        for file in source:
            name = file.split("/")[0]
            if name == file:
                shutil.copy(file, args.result + "/code_file/")
            else:
                os.makedirs(args.result + "/code_file/" + name, exist_ok=True)
                shutil.copy(file, args.result + "/code_file/" + name)

        for epoch in range(args.start_epoch, args.epochs):
            lr = adjust_learning_rate(optimizer, epoch, args, [1000, 2000])
            writer.add_scalar("lr", lr, epoch)

            # # train for one epoch
            loss = train(train_loader, model, lemniscate, criterion, optimizer, epoch, writer)
            writer.add_scalar("train_loss", loss, epoch)


            # save checkpoint
            if epoch % 1000 == 0 or (epoch in [1600, 1800, 2000]):
                auc, acc, precision, recall, f1score = kNN(args, model, lemniscate, train_loader, val_loader, 100,
                                                           args.nce_t, 2)
                # save to txt
                f = open(args.result+"/result.txt","a+")
                f.write("epoch " + str(epoch) + "\n")
                f.write("auc: %.4f\n" % (auc))
                f.write("acc: %.4f\n" % (acc))
                f.write("pre: %.4f\n" % (precision))
                f.write("recall: %.4f\n" % (recall))
                f.write("f1score: %.4f\n" % (f1score))
                f.close()
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'lemniscate': lemniscate,
                    'optimizer' : optimizer.state_dict(),
                }, filename = args.result + "/fold" +str(args.seedstart)+"-epoch-" +str(epoch) + ".pth.tar")


def train(train_loader, model, lemniscate, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_ins = AverageMeter()
    losses_rot = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (input, target, index, name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if args.synthesis:
            dataX = torch.cat(input,0).cuda()
            ori_data = dataX[:int(dataX.shape[0]/2)]
            syn_data = dataX[int(dataX.shape[0]/2):]
            data = [ori_data, syn_data]
            dataX = torch.stack(data, dim=1).cuda()
            batch_size, types, channels, height, width = dataX.size()
            input = dataX.view([batch_size * types, channels, height, width])

            # instance discrimination
            # input = torch.cat(input, 0).cuda()
            feature = model(input)
            loss = criterion(feature, index) / args.iter_size
        elif args.multiaug:

            input = torch.cat(input, 0).cuda()
            feature = model(input)
            loss = criterion(feature, index) / args.iter_size
        else:
            # instance discrimination memory bank
            input = input.cuda()
            index = index.cuda()

            feature = model(input)
            output = lemniscate(feature, index)
            loss = criterion(output, index) / args.iter_size

        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input.size(0))

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    writer.add_scalar("losses_ins", losses_ins.avg, epoch)
    writer.add_scalar("losses_rot", losses_rot.avg, epoch)

    return losses.avg



if __name__ == '__main__':
    main()
