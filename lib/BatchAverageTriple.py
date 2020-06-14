import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

def deleteFrom2D(arr2D, row, column):
    'Delete element from 2D numpy array by row and column position'
    arr2D = arr2D.cpu().data.numpy()
    modArr = np.delete(arr2D, row * arr2D.shape[1] + column)
    modArr = torch.from_numpy(modArr).cuda()
    return modArr

class BatchCriterionTriple(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, negM, T, batchSize, args):
        super(BatchCriterionTriple, self).__init__()
        self.negM = negM
        self.T = T
        self.domain = args.domain
        # self.stripe = stripe
        num = 3 if self.domain else 2
        self.diag_mat = 1 - torch.eye(batchSize * num).cuda()

        self.sigma = 4

        # self.m = 0.25
        self.m = 0.1
        self.gamma = 32
        self.soft_plus = nn.Softplus()

        self.mse = nn.MSELoss()

    def forward(self, x, targets):
        batchSize = x.size(0)

        # get positive innerproduct
        # print ("x", x[:,:5])

        reordered_x = torch.cat((x.narrow(0, int(batchSize // 3), int(batchSize // 3)), \
                                 x.narrow(0, int(2 * batchSize // 3), int(batchSize // 3)),
                                x.narrow(0, 0, int(batchSize // 3))), 0)

        # print ("reord", reordered_x[:,:5])
        #
        # idx = list(np.arange(1, int(batchSize), 2))
        # idx1 = np.array([item - 1 for item in idx])
        # # idx2 = np.array([item + 2 for item in idx])
        # # idx3 = np.array([item - 1 for item in idx])
        # idx = np.array(idx)
        # index = np.stack([idx, idx1])
        # index = list(index.flatten("F"))
        # reordered_x = reordered_x[index,:]

        # elif i == 2:
        #     idx = list(np.arange(2, int(batchSize), 4))
        #     idx1 = np.array([item + 1 for item in idx])
        #     idx2 = np.array([item - 2 for item in idx])
        #     idx3 = np.array([item - 1 for item in idx])
        #     idx = np.array(idx)
        #     index = np.stack([idx, idx1, idx2, idx3])
        #     index = list(index.flatten("F"))
        #     reordered_x = reordered_x[index,:]
        # elif i == 3:
        #     idx = list(np.arange(3, int(batchSize), 4))
        #     idx1 = np.array([item - 3 for item in idx])
        #     idx2 = np.array([item - 2 for item in idx])
        #     idx3 = np.array([item - 1 for item in idx])
        #     idx = np.array(idx)
        #     index = np.stack([idx, idx1, idx2, idx3])
        #     index = list(index.flatten("F"))
        #     reordered_x = reordered_x[index,:]

        # reordered_x = reordered_x.data

        pos_dist = (x * reordered_x.data).sum(1)
        pos = pos_dist.div_(self.T).exp_()

        weight = torch.ones_like(pos).cuda()
        weight[:int(batchSize // 3)] = self.sigma
        pos = pos * weight

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat
        weight_matric = torch.ones_like(all_prob).cuda()
        weight_matric[:int(batchSize//3),:] = self.sigma
        all_prob = all_prob * weight_matric


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
        #
        # print ("loss", loss)
        # exit(0)
        # # # # regularization loss
        # dist = (x * reordered_x.data).sum(1)
        # dist_ot = dist[: int(batchSize//3)]
        # dist_ts = dist[int(batchSize//3):2*int(batchSize)//3]
        # dist_os = dist[2*int(batchSize)//3:]
        #
        # print (dist_ot[:10])
        # print (dist_ts[:10])
        # print (dist_os[:10])
        # print (loss)
        #
        # print ("sp", dist)
        # all_prob = torch.mm(x, x.t().data)
        # print ("sn", all_prob)
        # exit(0)


        # # loss_mse = torch.sqrt(self.mse(dist_os,dist_ts) + 1e-6)
        # # print(loss_mse)
        # #
        # sp = dist_ot
        # sn = dist_ts
        # sn2 = dist_os
        # ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        # an = torch.clamp_min(sn.detach() + self.m, min=0.)
        # an2 = torch.clamp_min(sn2.detach() + self.m, min=0.)
        #
        # delta_p = 1 - self.m
        # delta_n = self.m
        #
        # logit_p = - ap * (sp - delta_p) * self.gamma
        # logit_n = an * (sn - delta_n) * self.gamma
        # logit_n2 = an2 * (sn2 - delta_n) * self.gamma
        #
        #
        # loss_reg = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        # loss_reg2 = self.soft_plus(torch.logsumexp(logit_n2, dim=0) + torch.logsumexp(logit_p, dim=0))
        #
        #
        # print(loss_reg, loss_reg2)
        # exit(0)
        # return loss + 0.1*loss_reg + 0.1*loss_reg2
        return loss
