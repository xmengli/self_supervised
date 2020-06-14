import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np



def deleteFrom2D(arr2D, row, column):
    'Delete element from 2D numpy array by row and column position'
    arr2D = arr2D.cpu().data.numpy()
    modArr = np.delete(arr2D, row * arr2D.shape[1] + column)
    modArr = torch.from_numpy(modArr).cuda()
    return modArr

class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, negM, T, batchSize, args):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.multitask = args.multitask
        self.domain =args.domain

        num = 2
        self.diag_mat = 1 - torch.eye(batchSize * num).cuda()

    def forward(self, x, targets):
        batchSize = x.size(0)

        # get positive innerproduct
        reordered_x = torch.cat((x.narrow(0, batchSize // 2, batchSize // 2), \
                                 x.narrow(0, 0, batchSize // 2)), 0)
        # reordered_x = reordered_x.data
        pos = (x * reordered_x.data).sum(1).div_(self.T).exp_()

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat


        if self.negM == 1:
            all_div = all_prob.sum(1)
        else:
            # remove pos for neg
            all_div = (all_prob.sum(1) - pos) * self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize, 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum) / batchSize


        # # triplet loss
        # cosdict = torch.mm(x, x.t().data)
        # cosdict = torch.ones_like(cosdict) - cosdict
        # # find postive index
        # rows = np.repeat(np.arange(0, cosdict.shape[0]), cosdict.shape[0] / 4)
        # columns = np.tile(np.reshape(np.arange(0, cosdict.shape[0]), (-1, 4)).flatten("F"), int(cosdict.shape[0] / 4))
        #
        # positive = cosdict[rows, columns]
        # negative = deleteFrom2D(cosdict, rows, columns)
        # positive = torch.repeat_interleave(positive, repeats=3)
        #
        # #  intra channel num should be smaller than inter channel.
        # distance_positive = positive
        # distance_negative = negative
        #
        # losses = F.relu(distance_positive - distance_negative + 0.1)
        # branch_loss = losses.mean()

        return loss
