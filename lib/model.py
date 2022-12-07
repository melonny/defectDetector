import os
from collections import OrderedDict
from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F

from lib.loss import ContrastiveLoss
from lib.networks import weights_init, NetG, VGG
from lib.visualizer import Visualizer
import torchvision.utils as vutils
from lib.utils import imshow
from scipy.stats import norm


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
        # self.device = 'cpu'
    ##
    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])


            # Copy the first batch as the fixed input.
            # if self.total_steps == self.opt.batchsize:
            #     self.fixed_input.resize_(input[0].size()).copy_(input[0])

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
        return OrderedDict([('err', self.err_g.item())])

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        # fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes

    def load_weights(self, epoch=0, is_best: bool = False, path=None):
        """ Load pre-trained weights of NetG and NetD
        Keyword Arguments:
            epoch {int}     -- Epoch to be loaded  (default: {None})
            is_best {bool}  -- Load the best epoch (default: {False})
            path {str}      -- Path to weight file (default: {None})
        Raises:
            Exception -- [description]
            IOError -- [description]
        """

        if epoch is None and is_best is False:
            raise Exception('Please provide epoch to be loaded or choose the best epoch.')

        if is_best:
            fname_g = f"netG_best.pth"
            # fname_d = f"netD_best.pth"
        else:
            fname_g = f"netG_{epoch}.pth"
            # fname_d = f"netD_{epoch}.pth"

        if path is None:
            path_g = f"{fname_g}"
            # path_g = f"/S21xul/ganomaly/ganomaly/ganomaly/output_9/ganomaly/train/weights/{fname_g}"
            # path_d = f"/S21xul/ganomaly/ganomaly/ganomaly/output_5/skipganomaly/train/weights/{fname_d}"

        # Load the weights of netg and netd.
        print('>> Loading weights...')
        # weights_g = torch.load(path_g, map_location='cpu')['state_dict']
        weights_g = torch.load(path_g)['state_dict']
        try:
            self.netg.load_state_dict(weights_g)
            # self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')

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
            # self.input = self.input.half()
            # print(data[0].shape)
            # print(self.input.shape)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes)

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

    def test(self):
        with torch.no_grad():
            # Load the weights of netg and netd.
            # if self.opt.load_weights:
            if True:
                path = "netG_40.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # print("   Testing model %s." % self.name)
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)

                        # Measure inference time.

                        # Scale error vector between [0, 1]


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
        # self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.netg.apply(weights_init)

        ##
        if self.opt.resume == '':
            print("\nLoading pre-trained networks.")
            # self.netg.half()
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG_40.pth'), map_location='cpu')['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG_40.pth'), map_location='cpu')['state_dict'])
            print("\tDone.\n")

        self.l_con = nn.L1Loss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)

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
        # self.device = "cpu"
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
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.00005)  # 定义优化器

        self.visualizer = Visualizer(opt)

        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = 41
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'net_40.pth'))['state_dict'])
            print("\tDone.\n")

    # def get_errors(self):
    #     """ Get netD and netG errors.
    #     Returns:
    #         [OrderedDict]: Dictionary containing errors.
    #     """
    #     errors = OrderedDict([
    #         ('err', self.err.item()),
    #         ])
    #
    #     return errors

    def train(self):
        criterion = ContrastiveLoss()  # 定义损失函数

        for epoch in range(0, self.opt.niter):
            torch.cuda.empty_cache()
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
        self.opt.phase = 'test'
        path = "output/aesiamese/train/weights/net_40.pth"
        pretrained_dict = torch.load(path, map_location='cpu')['state_dict']

        try:
            self.net.load_state_dict(pretrained_dict)
        except IOError:
            raise IOError("net weights not found")
        print('   Loaded net weights.')

        with torch.no_grad():
            self.ae.load_weights(epoch=40)
            # norm_error = 0
            # abnorm_error = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                img = data[0]
                label = data[1]
                self.ae.set_input(data)
                self.ae.fake, _, _ = self.ae.netg(self.ae.input)
                output1, output2 = self.net(img, self.ae.fake)
                o_euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
                euclidean_distance = torch.mean(o_euclidean_distance) * 16
                concatenated = torch.cat((img, self.ae.fake), 0)
                imshow(torchvision.utils.make_grid(concatenated),
                       'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
                # mu = 1968.2840576171875
                # sigma = 829.4754028320312
                #
                # threshould = mu + 2 * sigma
                # if(euclidean_distance > threshould):
                #     pred = 1
                # else:
                #     pred = 0
                # if(pred != label.item() and label == 0): norm_error += 1
                # if(pred != label.item() and label == 1): abnorm_error +=1
                # print('iter: {}, norm_error: {}'.format(i, norm_error))
                # print('iter: {}, abnorm_error: {}'.format(i, abnorm_error))

            #     if (dis < 0.5):
            #         guess = 0
            #     else:
            #         guess = 1
            #     if (guess != label):
            #         error += 1
            # total_error = error / len(self.dataloader['test'].dataset)
            # print(total_error)

    def aetrain(self):
        with torch.no_grad():
            self.ae.load_weights(epoch=40)
        self.net.train()
        criterion = ContrastiveLoss()  # 定义损失函数
        torch.cuda.empty_cache()
        for self.epoch in range(self.opt.iter, self.opt.niter):
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['train'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                img = data[0]
                label = data[1]
                # print(img.shape)
                self.ae.set_input(data)
                with torch.no_grad():
                    self.ae.fake, _, _ = self.ae.netg(self.ae.input)
                img, self.ae.fake, label = img.cuda(), self.ae.fake.cuda(), label.cuda()  # 数据移至GPU
                self.optimizer.zero_grad()
                output1, output2 = self.net(img, self.ae.fake)
                loss_contrastive = criterion(output1, output2, label)
                if self.total_steps % self.opt.print_freq == 0:
                    errors = OrderedDict([('err', loss_contrastive.item())])
                    if self.opt.display:
                        counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                        self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)
                loss_contrastive.backward()
                self.optimizer.step()
            print("Epoch number: {} , Current loss: {:.4f}\n".format(self.epoch, loss_contrastive.item()))
            if (self.epoch % 5 == 0):
                weight_dir = os.path.join(self.opt.outf, self.opt.model, 'train', 'weights')
                if not os.path.exists(weight_dir): os.makedirs(weight_dir)
                torch.save({'epoch': self.epoch, 'state_dict': self.net.state_dict()}, f"{weight_dir}/net_{self.epoch}.pth")

    def aeGaussion(self):
        self.opt.phase = 'test'
        # path = "output/aesiamese/train/weights/net_8.pth"
        # pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
        #
        # try:
        #     self.net.load_state_dict(pretrained_dict)
        # except IOError:
        #     raise IOError("net weights not found")
        # print('   Loaded net weights.')

        with torch.no_grad():
            self.ae.load_weights(epoch=40)
            a = []
            for i, data in enumerate(self.dataloader['train'], 0):
                img = data[0]
                label = data[1]
                self.ae.set_input(data)
                self.ae.fake, _, _ = self.ae.netg(self.ae.input)
                o_euclidean_distance = F.pairwise_distance(self.ae.input, self.ae.fake, keepdim=True)
                euclidean_distance = torch.sum(o_euclidean_distance)
                euclidean_distance.unsqueeze_(0)
                if(i == 0): a = euclidean_distance
                else: a = torch.cat((a, euclidean_distance),dim=0)
                print('iter: {}, dis: {}'.format(i, euclidean_distance))
            x = np.array(a)
            mu = np.mean(x)
            sigma = np.std(x)
            print(mu.item())
            print(sigma.item())
            num_bins = 30
            n,bins,patches = plt.hist(x, num_bins, density=1, alpha = 0.75)
            y = norm.pdf(bins, mu, sigma)
            plt.grid(True)
            plt.plot(bins, y, 'r--')
            plt.xlabel('values')
            plt.ylabel('prob')
            plt.show()
