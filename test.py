import torch
import time
import datasets
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np
from lib.utils import evaluation_metrics
import random
import os

my_whole_seed = 111
random.seed(my_whole_seed)
np.random.seed(my_whole_seed)
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(my_whole_seed)

def kNN(args, net, lemniscate, trainloader, testloader, K, sigma, C):
    net.eval()
    net_time = AverageMeter()


    trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    trainnames = []
    if args.multiaug:
    #     recomupte
        ndata = trainloader.dataset.__len__()
        trainFeatures = np.zeros((128, ndata))
        with torch.no_grad():
            transform_bak = trainloader.dataset.transform
            if args.saveembed:
                trainloader.dataset.train = True
                num = 50
            else:
                trainloader.dataset.transform = testloader.dataset.transform
                trainloader.dataset.train = False
                num = 100
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=num, shuffle=False, num_workers=4, worker_init_fn=random.seed(111))
            for batch_idx, (inputs, _, targets, indexes) in enumerate(temploader):
                if args.saveembed:
                    inputs = torch.cat(inputs, 0).cuda()
                    # ori_data = dataX[:int(dataX.shape[0] / 2)]
                    # syn_data = dataX[int(dataX.shape[0] / 2):]
                    # data = [ori_data, syn_data]
                    # dataX = torch.stack(data, dim=1).cuda()
                    # batch_size, types, channels, height, width = dataX.size()
                    # inputs = dataX.view([batch_size * types, channels, height, width])
                batchSize = inputs.size(0)

                if args.multitask and args.domain:
                    features_inst, features_rot = net(inputs)
                    trainFeatures[:,batch_idx * batchSize:batch_idx * batchSize + batchSize] = features_inst.data.t().cpu().numpy()
                elif args.multitask:
                    features_inst, features_rot, features = net(inputs)
                    trainFeatures[:,batch_idx * batchSize:batch_idx * batchSize + batchSize] = features_inst.data.t().cpu().numpy()
                else:
                    features = net(inputs)
                    trainFeatures[:,batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t().cpu().numpy()

                trainnames += list(indexes)
        trainloader.dataset.transform = transform_bak
        trainloader.dataset.train = True
        trainFeatures = torch.Tensor(trainFeatures).cuda()
    else:
        trainFeatures = lemniscate.memory.t()


    pred_box = []
    label_box = []

    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes, name) in enumerate(testloader):

            end = time.time()
            targets = targets.cuda()
            batchSize = inputs.size(0)
            if args.multitask and args.domain:
                features, features_rot = net(inputs)
            elif args.multitask:
                features, features_rot, features_whole = net(inputs)
            else:
                features = net(inputs)
            net_time.update(time.time() - end)

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)

            # print ("yi", yi[1])
            # idx = yi[1].cpu().data.numpy()
            # idx = list(idx)
            # print ("idx", idx)
            # print (trainnames)
            # print ("train", [trainnames[item] for item in idx])
            # dist = dist[1].cpu().data.numpy()
            # print ("dist", [dist[item] for item in idx])
            # exit(0)

            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)


            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # get pred result
            pred = predictions.narrow(1,0,1).cpu().numpy()
            pred = [item[0] for item in pred]

            # append to result box
            pred_box += pred
            target_numpy = list(targets.cpu().numpy())
            label_box += target_numpy


    auc, acc, precision, recall, f1score = evaluation_metrics(label_box, pred_box, C)


    return auc, acc, precision, recall, f1score

