import os
import sys
import math
import numpy as np
# import cv2
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean() / 2 + 0.5
    else:
        return ssim_map / 2 + 0.5


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class PatchLoss(nn.Module):
    def __init__(self, patch_size=64, n_patches=4):
        super(PatchLoss, self).__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches

    def forward(self, img1, img2):
        b, c, h, w = img1.shape
        img = torch.cat([img1, img2], dim=0)

        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.reshape(2 * b, c, -1, self.patch_size, self.patch_size).permute(2, 0, 1, 3, 4)
        total_patches = patches.size(0)

        index = torch.randint(patches.size(0), (self.n_patches,))
        patches = patches[index]

        patch1 = patches[:, :b]
        patch2 = patches[:, b:]
        loss = nn.MSELoss()(patch1, patch2)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, self.n_patches)
        # for i in range(self.n_patches):
        #     show_img1 = patch1[i, 0, 0].reshape(self.patch_size, self.patch_size).detach().cpu().numpy()
        #     ax[0][i].imshow(show_img1, cmap='gray')
        #     ax[0][i].axis('off')
        #     show_img2 = patch2[i, 0, 0].reshape(self.patch_size, self.patch_size).detach().cpu().numpy()
        #     ax[1][i].imshow(show_img2, cmap='gray')
        #     ax[1][i].axis('off')
        # plt.show()

        return loss

class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'sum', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}.')

    def forward(self, pred, target):
        bp, cp, _, _ = pred.shape
        bt, ct, _, _ = target.shape

        assert cp == ct and bp == bt
        b = bp

        kx = torch.Tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)
        # kx = kx.repeat((3, 1, 1, 1))
        # ky = ky.repeat((3, 1, 1, 1))

        loss = 0

        for i in range(cp):
            p = pred[:, i, :, :].unsqueeze(1)
            t = target[:, i, :, :].unsqueeze(1)

            pred_grad_x = F.conv2d(p, kx, padding=1)
            pred_grad_y = F.conv2d(p, ky, padding=1)
            target_grad_x = F.conv2d(t, kx, padding=1)
            target_grad_y = F.conv2d(t, ky, padding=1)

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(2, 4)
            # pdx = pred_grad_x[0][0].detach().cpu().numpy()
            # pdy = pred_grad_x[0][0].detach().cpu().numpy()
            # tdx = target_grad_x[0][0].cpu().numpy()
            # tdy = target_grad_y[0][0].cpu().numpy()
            # ax[0][0].imshow(pred[0][0].detach().cpu().numpy(), cmap='gray'); ax[0][0].set_title('pred image')
            # ax[0][1].imshow(pdx, cmap='gray'); ax[0][1].set_title('pred dx')
            # ax[0][2].imshow(pdy, cmap='gray'); ax[0][2].set_title('pred dy')
            # ax[0][3].imshow(pdx + pdy, cmap='gray'); ax[0][3].set_title('pred dx+dy')
            # ax[1][0].imshow(target[0][0].cpu().numpy(), cmap='gray'); ax[1][0].set_title('gt image')
            # ax[1][1].imshow(tdx, cmap='gray'); ax[1][1].set_title('gt dx')
            # ax[1][2].imshow(tdy, cmap='gray'); ax[1][2].set_title('gt dy')
            # ax[1][3].imshow(tdx + tdy, cmap='gray'); ax[1][3].set_title('gt dx+dy')
            # plt.show()

            x_loss = nn.L1Loss(reduction=self.reduction)(pred_grad_x, target_grad_x)
            y_loss = nn.L1Loss(reduction=self.reduction)(pred_grad_y, target_grad_y)

            loss += x_loss + y_loss

        return loss * self.loss_weight


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # if input.min() < 0:
        #     input = input * 0.5 + 0.5
        #     target = target * 0.5 + 0.5
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class SimilarityLoss(nn.Module):
    def __init__(self, resize=True):
        super(SimilarityLoss, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True).eval()

        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        x = self.resnet(input)
        y = self.resnet(target)

        # distance = F.pairwise_distance(x, y)
        distance = self.cos_sim(x, y)
        return distance.mean()

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class LandmarksLoss(nn.Module):
    def __init__(self):
        super(LandmarksLoss, self).__init__()

        import face_alignment
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.n_lndm = 68

    def forward(self, pred, target):

        device = pred.device
        h, w = pred.shape[2:]

        target = (target * 0.5 + 0.5) * 255.
        target = target.repeat(1, 3, 1, 1)
        target_landmarks = self.fa.get_landmarks_from_batch(target)

        pred = (pred * 0.5 + 0.5) * 255.
        pred_landmarks = self.fa.get_landmarks_from_batch(pred)

        loss = 0
        for i in range(pred.size(0)):

            tar_lndm = torch.tensor(target_landmarks[i][:self.n_lndm]).to(device)
            tar_lndm[:, 0] /= w
            tar_lndm[:, 1] /= h

            pred_lndm = torch.tensor(pred_landmarks[i][:self.n_lndm]).to(device)
            if len(pred_lndm) == 0:
                pred_lndm = torch.zeros_like(tar_lndm)
            pred_lndm[:, 0] /= w
            pred_lndm[:, 1] /= h

            loss += nn.MSELoss()(pred_lndm, tar_lndm)

        return loss.mean()


