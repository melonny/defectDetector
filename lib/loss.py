import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        my_loss = nn.CrossEntropyLoss()
        o_euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        euclidean_distance = torch.mean(o_euclidean_distance) * 16
        # print(output1.shape)
        # print(output2.shape)
        # print(euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # loss = my_loss(output3, label.squeeze(1).long()) +  loss_contrastive
        loss = loss_contrastive
        return loss