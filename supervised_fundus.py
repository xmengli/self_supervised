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
from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score
from lib.utils import AverageMeter
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
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
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
parser.add_argument('--seedend', default=5, type=int)


parser.add_argument("--synthesis", action="store_true")
parser.add_argument('--showfeature', action="store_true")
parser.add_argument('--multiaug', action="store_true")
parser.add_argument('--multitask', action="store_true")
parser.add_argument("--multitaskposrot", action="store_true")
best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()

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

    for kk_time in range(args.seedstart, args.seedend):
        args.seed = kk_time
        args.result = args.result + str(args.seed)

        # create model
        from models.resnet_sup import resnet18, resnet50, resnet34
        model = resnet34()


        model = torch.nn.DataParallel(model).cuda()


        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        aug = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                                  # transforms.RandomGrayscale(p=0.2),
                                  # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize])
        # aug = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.), ratio=(3 / 4, 4 / 3)),
        #                           transforms.RandomHorizontalFlip(p=0.5),
        #                           get_color_distortion(s=1),
        #                           transforms.Lambda(lambda x: gaussian_blur(x)),
        #                           transforms.ToTensor(),
        #                           normalize])
        # aug = transforms.Compose([transforms.RandomRotation(60),
        #                           transforms.RandomResizedCrop(224, scale=(0.6, 1.)),
        #                           transforms.RandomGrayscale(p=0.2),
        #                           transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #                           transforms.RandomHorizontalFlip(),
        #                           transforms.ToTensor(),
        #                             normalize])
        aug_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize])

        # dataset
        import datasets.fundus_amd_crossvalidation as medicaldata
        train_dataset = medicaldata.traindataset(root=args.data, transform=aug, train=True, args=args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True if args.multiaug else False,  worker_init_fn=random.seed(my_whole_seed))


        valid_dataset = medicaldata.traindataset(root=args.data, transform=aug_test, train=False, args=args)
        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4,
            worker_init_fn=random.seed(my_whole_seed))

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model_dict = model.state_dict()

                pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict}
                pretrained_dict.pop("module.fc.weight")
                pretrained_dict.pop("module.fc.bias")
                # pretrained_dict = {k: v for k, v in checkpoint["net"].items() if k in model_dict}
                # pretrained_dict.pop("module.conv1.weight")
                # pretrained_dict.pop("module.conv1.bias")

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))

            else:
                print("=> no checkpoint found at '{}'".format(args.resume))


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
            lr = adjust_learning_rate(optimizer, epoch, args, [500, 1000, 1500])
            writer.add_scalar("lr", lr, epoch)

            # # train for one epoch
            loss = train(train_loader, model, criterion, optimizer)
            writer.add_scalar("train_loss", loss, epoch)

            gap_int = 200
            if (epoch) % gap_int == 0:
                loss_val, auc, acc, precision, recall, f1score = supervised_evaluation(model, val_loader)
                writer.add_scalar("test_auc", auc, epoch)
                writer.add_scalar("test_acc", acc, epoch)
                writer.add_scalar("test_precision", precision, epoch)
                writer.add_scalar("test_recall", recall, epoch)
                writer.add_scalar("test_f1score", f1score, epoch)


                # save to txt
                f = open(args.result+"/result.txt","a+")
                f.write("epoch " + str(epoch) + "\n")
                f.write("auc: %.4f\n" % (auc))
                f.write("acc: %.4f\n" % (acc))
                f.write("pre: %.4f\n" % (precision))
                f.write("recall: %.4f\n" % (recall))
                f.write("f1score: %.4f\n" % (f1score))
                f.close()


                # save checkpoint
            if epoch in [1000,2000,3000]:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, filename = args.result + "/epoch-" +str(epoch) + ".pth.tar")



def supervised_evaluation(model, val_loader):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.train()

    prediction_box = []
    target_box = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target, index, name) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            output = model(images)

            output = torch.softmax(output,dim=1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)

            prediction_box += list(output)
            target_box += list(target.data.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    auc = roc_auc_score(target_box, prediction_box)
    corrects = np.equal(np.array(target_box), np.array(prediction_box))
    acc = float(sum(corrects)) / len(corrects)

    #  mean class
    precision = precision_score(target_box, prediction_box, average='macro')
    recall = recall_score(target_box, prediction_box, average='macro')
    f1score = f1_score(target_box, prediction_box, average='macro')

    return losses.avg, round(auc, 4), round(acc, 4), round(precision, 4),  round(recall, 4),  round(f1score, 4)


def train(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (input, target, index, name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()


        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


if __name__ == '__main__':
    main()
