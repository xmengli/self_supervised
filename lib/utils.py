import torch
import shutil
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
# import cv2
from skimage.transform import resize
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch, args, interval):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr
    if epoch < interval[0]:
        lr = args.lr
    elif epoch >= interval[0] and epoch < interval[1]:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    #lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def multi_class_auc(all_target, all_output, num_c = None):
    from sklearn.preprocessing import label_binarize

    # all_output = np.stack(all_output)
    all_target = label_binarize(all_target, classes=list(range(0, num_c)))
    all_output = label_binarize(all_output, classes=list(range(0, num_c)))
    auc_sum = []

    for num_class in range(0, num_c):
        try:
            auc = roc_auc_score(all_target[:, num_class], all_output[:, num_class])
            auc_sum.append(auc)
        except ValueError:
            pass

    auc = sum(auc_sum) / (float(len(auc_sum))+1e-8)

    return auc

def evaluation_metrics(label, pred, C):

    if C==2:
        auc = roc_auc_score(label, pred)
    else:
        auc = multi_class_auc(label, pred, num_c=C)
    
    corrects = np.equal(np.array(label), np.array(pred))
    acc = float(sum(corrects)) / len(corrects)

    #  mean class
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1score = f1_score(label, pred, average='macro')

    return round(auc, 4), round(acc, 4), round(precision, 4), round(recall, 4), round(f1score, 4)



def showfeature(x, savename):
    # trun to numpy
    x = x.data.cpu().numpy()
    print (x.shape)
    box = []
    for item in range(0, x.shape[0]):
        x_patch = x[item, :, :]
        box.append(x_patch)
    x_patch = np.stack(box)
    x_patch = np.max(x_patch, axis=0)
    x_patch = resize(x_patch, (224, 224), order=3, mode='constant',
                     cval=0, clip=True, preserve_range=True)
    x_patch = (x_patch - np.min(x_patch)) / (np.max(x_patch) - np.min(x_patch) + 1e-11)
    x_patch = x_patch * 255
    x_patch = np.array(x_patch, dtype="uint8")
    plt.plot(1), plt.imshow(x_patch, cmap='jet')
    plt.axis('off')
    plt.savefig(savename, bbox_inches='tight', pad_inches=0)

def showimage(x, savename):
    import torchvision.transforms as transforms

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    z = x * torch.tensor(std).view(3, 1, 1).cuda()
    z = z + torch.tensor(mean).view(3, 1, 1).cuda()
    z = z.cpu()
    z = z[[2,1,0], : ,:]
    img2 = transforms.ToPILImage()(z)
    img2.save(savename)

def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    return color_distort


def gaussian_blur(x):
    from PIL.ImageFilter import GaussianBlur
    if np.random.randint(0, 2) == 1:
        x = x.filter(GaussianBlur(radius=np.random.uniform(0.1, 2.0)))
    return x
