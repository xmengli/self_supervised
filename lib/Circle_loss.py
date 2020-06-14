from typing import Tuple

import torch
from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        self.diag_mat = 1 - torch.eye(100 * 3).cuda()
        self.T = 0.1


    def forward(self, x: Tensor, targets: Tensor) -> Tensor:

        # cal sp, sn from x
        batchSize = x.size(0)
        reordered_x = torch.cat((x.narrow(0, int(batchSize // 3), int(batchSize // 3)), \
                                 x.narrow(0, int(2 * batchSize // 3), int(batchSize // 3)),
                                x.narrow(0, 0, int(batchSize // 3))), 0)


        # # # regularization loss

        pos = (x * reordered_x.data).sum(1).div_(self.T).exp_()

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat

        all_div = all_prob.sum(1)


        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize, 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        sp = lnPmt
        sn = lnPon
        lnPon = lnPon.log_()
        lnPmt = lnPmt.log_()
        # exit(0)
        # # ########################################  add alpha, beta
        ap = torch.clamp_min(- lnPmt.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(lnPon.detach() + self.m, min=0.)
        #
        delta_p = 1 - self.m
        delta_n = self.m
        #
        lnPmt = - ap * (sp - delta_p) * self.gamma
        lnPon = an * (sn - delta_n) * self.gamma

        lnPmt = lnPmt.exp_()
        lnPon = lnPon.exp_()
#####################################################

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)
        print (lnPmtsum, lnPonsum)
        loss = - (lnPmtsum + lnPonsum) / batchSize
        print ("loss", loss)
        exit(0)

        sp = lnPmt
        sn = lnPon


        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        # logit_p = - ap * (sp - delta_p) * self.gamma
        # logit_n = an * (sn - delta_n) * self.gamma

        logit_p = -sp
        logit_n = sn

        print (logit_n)
        print (logit_p)

        print (logit_n.shape)
        print (logit_p.shape)
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        print ("loss", loss/300)
        exit(0)
        # return loss.mean()
