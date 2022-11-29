import os
from collections import OrderedDict
from datetime import time

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F

from lib.loss import ContrastiveLoss
from lib.networks import weights_init, NetG, VGG
from lib.visualizer import Visualizer


class BaseModel():
    """ Base Model for ganomaly
    """

    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

    ##
    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        return self.err_g.item()

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/netG_{epoch}.pth")

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()
            errors = self.get_errors()
            if self.total_steps % self.opt.print_freq == 0:
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            self.save_weights(self.epoch)
            break
            # self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    # def test(self):
    #     with torch.no_grad():
    #         # Load the weights of netg and netd.
    #         if self.opt.load_weights!='':
    #             path = self.opt.load_weights.format(self.name.lower(), self.opt.dataset)
    #             pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
    #             try:
    #                 self.netg.load_state_dict(pretrained_dict)
    #             except IOError:
    #                 raise IOError("weights not found")
    #             print('   Loaded weights.')
    #         self.opt.phase = 'test'
    #         print("   Testing model %s." % self.name)
    #         self.times = []
    #         self.total_steps = 0
    #         epoch_iter = 0
    #         error = 0
    #         for i, data in enumerate(self.dataloader['test'], 0):
    #             self.total_steps += self.opt.batchsize
    #             epoch_iter += self.opt.batchsize
    #             time_i = time.time()
    #             self.set_input(data)
    #             self.fake, _, _ = self.netg(self.input)
    #             time_o = time.time()
    #             self.times.append(time_o - time_i)
    #             error = self.get_errors() + error



##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self):
        return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netg.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_con = nn.L1Loss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize, self.opt.isize),
                                       dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g = self.l_con(self.fake, self.input)
        self.err_g.backward()

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()


class SiameseModel():
    def __init__(self, opt, dataloader):
        # Initalize variables.
        self.opt = opt
        # self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.epoch = 0
        self.total_steps = 0
        ##
        # Create and initialize networks.
        self.net = VGG('VGG11').to(self.device)
        self.net.apply(weights_init)

        self.ae = Ganomaly(opt, dataloader)
        # pretrained_dict = torch.load(path)['state_dict']
        # self.ae.netg.load_state_dict(pretrained_dict)

        if self.opt.isTrain:
            self.net.train()
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.0005)  # 定义优化器

    def train(self):
        criterion = ContrastiveLoss()  # 定义损失函数
        for epoch in range(0, self.opt.niter):
            for i, data in enumerate(self.dataloader['train'], 0):
                img0, img1, label = data
                # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()  # 数据移至GPU
                self.optimizer.zero_grad()
                output1, output2 = self.net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                print(loss_contrastive.item())
                loss_contrastive.backward()
                self.optimizer.step()
            print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))
            if (epoch % 10 == 0):
                weight_dir = os.path.join(self.opt.outf, self.opt.model, 'train', 'weights')
                if not os.path.exists(weight_dir): os.makedirs(weight_dir)
                torch.save({'epoch': epoch, 'state_dict': self.net.state_dict()}, f"{weight_dir}/net_{epoch}.pth")
    def aetest(self):
        # ae = Ganomaly(self.opt, self.dataloader['train'])
        error = 0
        with torch.no_grad:
            for i, data in enumerate(self.dataloader['test'], 0):
                img = data[0]
                label = data[1]
                fake, _, _ = self.ae.netg(img)
                dis = F.pairwise_distance(img, label, keepdim=True)
                if(dis < 0.5):
                    guess=0
                else:
                    guess=1
                if(guess != label):
                    error +=1
            total_error = error / len(self.dataloader['test'].dataset)
            print(total_error)

    def aetrain(self):
        self.net.train()
        criterion = ContrastiveLoss()  # 定义损失函数
        for epoch in range(0, self.opt.niter):
            for i, data in enumerate(self.dataloader['train'], 0):
                img = data[0]
                label = data[1]
                fake, _, _= self.ae.netg(img)
                # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()  # 数据移至GPU
                self.optimizer.zero_grad()
                output1, output2 = self.net(img, fake)
                loss_contrastive = criterion(output1, output2, label)
                print(loss_contrastive.item())
                loss_contrastive.backward()
                self.optimizer.step()
            print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))
            if (epoch % 10 == 0):
                weight_dir = os.path.join(self.opt.outf, self.opt.model, 'train', 'weights')
                if not os.path.exists(weight_dir): os.makedirs(weight_dir)
                torch.save({'epoch': epoch, 'state_dict': self.net.state_dict()}, f"{weight_dir}/net_{epoch}.pth")
