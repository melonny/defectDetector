from __future__ import print_function

##
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from lib.dataloader import loadAEdata, loadSiamesedata, AEDataset
from lib.loss import ContrastiveLoss
from lib.model import Ganomaly, SiameseModel
from options import Options


def train():
    """ Training
    """

    ##
    # ARGUMENTS
    global model
    opt = Options().parse()

    if(opt.model == 'ganomaly'):
        dataloader = loadAEdata(opt)
        model = Ganomaly(opt, dataloader)

    if(opt.model == 'siamese' or opt.model == 'aesiamese'):
        dataloader = loadSiamesedata(opt)
        model = SiameseModel(opt, dataloader)

    ##
    # TRAIN MODEL
    model.train()

if __name__ == '__main__':
    train()