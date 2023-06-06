import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from utils.metrics import SSIM, VGGPerceptualLoss, GradientLoss, SimilarityLoss


class ShadowRemoval(nn.Module):
    def __init__(self, opt, train_mode=True):
        super(ShadowRemoval, self).__init__()
        self.opt = opt
        self.epoch = 0
        self.train_mode = train_mode

        self.light_channels = 27
        self.encoder = UNetEncoder(in_channels=1, out_channels=16 * 16 + self.light_channels, num_filter=16)
        self.generate_normal = UNetDecoder(in_channels=16 * 16, out_channels=3, num_filter=16)
        self.generate_sf = UNetDecoder(in_channels=16 * 16 + self.light_channels, out_channels=1, num_filter=16)
        self.lighting_net = LightingNet(in_channels=self.light_channels, out_channels=3)

        self.refine_net = GatedUNet(in_channels=2, out_channels=1)


        if train_mode:
            self.optimizer = optim.Adam(self.parameters(), lr=opt.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[8, 12],
                                                            gamma=0.5)

        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.gradient_loss = GradientLoss()
        self.ssim = SSIM()

        self.losses = {}


    def forward(self, data, **kwargs):
        img, albedo, normal, light, gt = data['input'], data['albedo'], data['normal'], data['light'], data['gt']

        img = img * 2 - 1
        gt = gt * 2 - 1

        x, features = self.encoder(img)

        normal_feat = x[:, :-self.light_channels, :, :]
        light_feat = x[:, -self.light_channels:, :, :]

        pred_normal = self.generate_normal(normal_feat, features)

        pred_light = self.lighting_net.down(light_feat)

        tar_light = torch.tensor([[0, 0, 1]]).repeat(img.size(0), 1).float().to(img.device)
        tar_light_feat = self.lighting_net.up(tar_light)
        tar_light_feat = tar_light_feat.repeat(1, 1, x.size(2), x.size(3))

        sf_feat = torch.cat([normal_feat, tar_light_feat], dim=1)

        ratio_unet = self.generate_sf(sf_feat, features)
        out_unet = img * ratio_unet

        shading = get_shading(pred_normal, pred_light)

        tar_shading = get_shading(pred_normal, tar_light)
        mask = tar_shading - shading
        mask = torch.clamp(mask, 0, 1)

        refine_mask = torch.cat([1 - (ratio_unet * 0.5 + 0.5), mask], dim=1)
        ratio_refine = self.refine_net(out_unet, refine_mask)
        out = img * ratio_refine

        light_loss = self.mseloss(pred_light, light)

        normal_loss = self.mseloss(pred_normal, normal)

        pix_loss = (self.l1loss(out, gt) + self.l1loss(out_unet, gt)) / 2

        smask = data['shadow_part_mask']
        shadow_loss = nn.L1Loss(reduction='sum')(out_unet * smask, gt * smask) / (smask > 0).sum()
        shadow_loss += nn.L1Loss(reduction='sum')(out * smask, gt * smask) / (smask > 0).sum()

        rmask = 1 - (ratio_unet * 0.5 + 0.5)
        refine_loss = nn.L1Loss(reduction='sum')(out * rmask, gt * rmask)
        refine_loss /= rmask.sum()

        perceptual_loss = (self.perceptual_loss(out, gt) + self.perceptual_loss(out_unet, gt)) / 2

        loss = light_loss + normal_loss + pix_loss + 0.1 * perceptual_loss + refine_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.update({
            'light': light_loss,
            'normal': normal_loss,
            'pix': pix_loss,
            'shadow': shadow_loss,
            'refine': refine_loss,
            'perceptual': perceptual_loss,
            'total': loss
        })

        return self.losses

    def predict(self, img, gt=None):
        with torch.no_grad():
            img = img * 2 - 1

            x, features = self.encoder(img)

            normal_feat = x[:, :-self.light_channels, :, :]
            light_feat = x[:, -self.light_channels:, :, :]

            pred_normal = self.generate_normal(normal_feat, features)

            pred_light = self.lighting_net.down(light_feat)

            tar_light = torch.tensor([[0, 0, 1]]).repeat(img.size(0), 1).float().to(img.device)
            tar_light_feat = self.lighting_net.up(tar_light)
            tar_light_feat = tar_light_feat.repeat(1, 1, x.size(2), x.size(3))

            sf_feat = torch.cat([normal_feat, tar_light_feat], dim=1)

            ratio_unet = self.generate_sf(sf_feat, features)
            out_unet = img * ratio_unet

            shading = get_shading(pred_normal, pred_light)

            tar_shading = get_shading(pred_normal, tar_light)
            mask = tar_shading - shading
            mask = torch.clamp(mask, 0, 1)

            refine_mask = torch.cat([1 - (ratio_unet * 0.5 + 0.5), mask], dim=1)
            ratio_refine = self.refine_net(out_unet, refine_mask)
            out = img * ratio_refine

            out = out * 0.5 + 0.5

            return out

    def visualize(self, vis, data, use_gt=True):
        img = data['input']
        vis.show_images(img, win='input')

        if use_gt:
            albedo, normal, light, gt = data['albedo'], data['normal'], data['light'], data['gt']
            vis.show_images(gt, win='GT')


        with torch.no_grad():

            img = img * 2 - 1

            x, features = self.encoder(img)

            normal_feat = x[:, :-self.light_channels, :, :]
            light_feat = x[:, -self.light_channels:, :, :]

            pred_normal = self.generate_normal(normal_feat, features)

            pred_light = self.lighting_net.down(light_feat)

            tar_light = torch.tensor([[0, 0, 1]]).repeat(img.size(0), 1).float().to(img.device)
            tar_light_feat = self.lighting_net.up(tar_light)
            tar_light_feat = tar_light_feat.repeat(1, 1, x.size(2), x.size(3))

            sf_feat = torch.cat([normal_feat, tar_light_feat], dim=1)

            ratio_unet = self.generate_sf(sf_feat, features)
            out_unet = img * ratio_unet

            shading = get_shading(pred_normal, pred_light)

            tar_shading = get_shading(pred_normal, tar_light)
            mask = tar_shading - shading
            mask = torch.clamp(mask, 0, 1)

            refine_mask = torch.cat([1 - (ratio_unet * 0.5 + 0.5), mask], dim=1)
            ratio_refine = self.refine_net(out_unet, refine_mask)
            out = img * ratio_refine

            if use_gt:
                gt_shading = get_shading(normal, light)
                gt_shading = (gt_shading - gt_shading.min()) / (gt_shading.max() - gt_shading.min())

                vis.show_images(normal * 0.5 + 0.5, win='gt_normal')
                vis.show_images(gt_shading, win='gt_shading')

            vis.show_images(out * 0.5 + 0.5, win='output')
            vis.show_images(pred_normal * 0.5 + 0.5, win='pred_normal')
            shading = (shading - shading.min()) / (shading.max() - shading.min())
            vis.show_images(shading, win='pred_shading')

            tar_shading = (tar_shading - tar_shading.min()) / (tar_shading.max() - tar_shading.min())
            vis.show_images(tar_shading, win='tar_shading')

            vis.show_images(mask, win='shadow_mask')
            vis.show_images(out_unet * 0.5 + 0.5, win='out_unet')

            vis.show_images(ratio_unet * 0.5 + 0.5, win='ratio_unet')
            vis.show_images(out_unet * 0.5 + 0.5, win='out_unet')
            vis.show_images(ratio_refine * 0.5 + 0.5, win='ratio_refine')
            vis.show_images(refine_mask.mean(dim=1, keepdims=True), win='refine_mask')
            vis.show_images(out * 0.5 + 0.5, win='out_refine')


    def save(self, epoch):
        self.epoch = epoch
        save_path = os.path.join(self.opt.checkpoints_dir, 'epoch_%s.pth' % epoch)
        checkpoints = {'epoch': epoch,
                       'model': self.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'model_name': self.opt.model}
        if self.use_D:
            checkpoints.update({
                'netD': self.netD.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict()
            })
        torch.save(checkpoints, save_path)
    #
    def load(self, load_path, load_optimizer=True):
        checkpoints = torch.load(load_path, map_location='cuda:0')
        print('Load model Epoch ' + str(checkpoints['epoch']))
        self.epoch = checkpoints['epoch'] + 1
        self.load_state_dict(checkpoints['model'])
        if load_optimizer and self.train_mode:
            self.optimizer.load_state_dict(checkpoints['optimizer'])
        return self.epoch

    def adjust_learning_rate(self):
        self.scheduler.step()
        if self.use_D:
            self.scheduler_D.step()
        lr = self.scheduler.get_lr()[0]
        return lr

    def load_for_refine(self, load_path):
        checkpoints = torch.load(load_path)
        pretrained_state_dict = checkpoints['model']
        refine_state_dict = self.state_dict()
        for rname, pname in zip(refine_state_dict.keys(), pretrained_state_dict.keys()):
            if rname == pname:
                refine_state_dict[rname] = pretrained_state_dict[pname]
        self.load_state_dict(refine_state_dict)



def get_shading(N, L):
    c1 = 0.8862269254527579
    c2 = 1.0233267079464883
    c3 = 0.24770795610037571
    c4 = 0.8580855308097834
    c5 = 0.4290427654048917

    nx = N[:, 0, :, :]
    ny = N[:, 1, :, :]
    nz = N[:, 2, :, :]

    b, c, h, w = N.shape

    Y1 = c1 * torch.ones(b, h, w).to(N.device)
    Y2 = c2 * ny
    Y3 = c2 * nz
    Y4 = c2 * nx
    Y5 = c4 * ny * nx
    Y6 = c4 * ny * nz
    Y7 = c3 * (3 * nz * nz - 1)
    Y8 = c4 * nx * nz
    Y9 = c5 * (nx * nx - ny * ny)

    if L.size(1) == 3:
        x, y, z = L.split(1, dim=-1)
        sh = torch.cat([torch.ones_like(x), y, z, x, y * x, y * z, 3 * z * z - 1, x * z, x * x - y * y], dim=-1)
    elif L.size(1) == 9:
        sh = L.clone()
    else:
        raise ValueError('Current light shape: {}, but need to shape: (B, 3) or (B, 9)'.format(L.size()))


    Y1 = Y1.to(N.device)
    l = sh  # [j]
    l = l.repeat(1, h * w).view(b, h, w, 9)
    l = l.permute([0, 3, 1, 2])
    shading = Y1 * l[:, 0] + Y2 * l[:, 1] + Y3 * l[:, 2] + \
              Y4 * l[:, 3] + Y5 * l[:, 4] + Y6 * l[:, 5] + \
              Y7 * l[:, 6] + Y8 * l[:, 7] + Y9 * l[:, 8]
    shading = shading.unsqueeze(1)

    return shading


def get_shading_AE(N, AE):
    device = N.device

    B, C, H, W = N.shape

    a, e = AE.split(1, -1)
    x = torch.cos(e) * torch.sin(-a)
    y = torch.sin(e)
    z = torch.cos(e) * torch.cos(-a)

    SH = torch.cat([torch.ones_like(x), y, z, x, y * x, y * z, 3 * z * z - 1, x * z, x * x - y * y], -1) * 0.7
    SH = SH.reshape(-1, 9, 1)

    N = N.reshape(B, C, -1)

    norm_X = N[:, 0, :]
    norm_Y = N[:, 1, :]
    norm_Z = N[:, 2, :]

    sh_basis = torch.zeros((B, H * W, 9)).to(device)
    att = torch.pi * torch.tensor([1, 2.0 / 3.0, 1 / 4.0]).to(device)
    sh_basis[:, :, 0] = 0.5 / (torch.pi ** 0.5) * att[0]

    sh_basis[:, :, 1] = (3 ** 0.5) / 2 / (torch.pi ** 0.5) * norm_Y * att[1]
    sh_basis[:, :, 2] = (3 ** 0.5) / 2 / (torch.pi ** 0.5) * norm_Z * att[1]
    sh_basis[:, :, 3] = (3 ** 0.5) / 2 / (torch.pi ** 0.5) * norm_X * att[1]

    sh_basis[:, :, 4] = (15 ** 0.5) / 2 / (torch.pi ** 0.5) * norm_Y * norm_X * att[2]
    sh_basis[:, :, 5] = (15 ** 0.5) / 2 / (torch.pi ** 0.5) * norm_Y * norm_Z * att[2]
    sh_basis[:, :, 6] = (5 ** 0.5) / 4 / (torch.pi ** 0.5) * (3 * norm_Z ** 2 - 1) * att[2]
    sh_basis[:, :, 7] = (15 ** 0.5) / 2 / (torch.pi ** 0.5) * norm_X * norm_Z * att[2]
    sh_basis[:, :, 8] = (15 ** 0.5) / 4 / (torch.pi ** 0.5) * (norm_X ** 2 - norm_Y ** 2) * att[2]

    shading = torch.bmm(sh_basis, SH)

    shading = shading.reshape(B, -1, H, W)

    return shading


def rgb2gray(rgb):
    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gradient(img):
    kx = torch.Tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]]).view(1, 1, 3, 3).to(img)
    ky = torch.Tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]]).view(1, 1, 3, 3).to(img)

    grad = []

    for i in range(img.size(1)):
        ci = img[:, i, :, :].unsqueeze(1)

        grad_x = F.conv2d(ci, kx, padding=1)
        grad_y = F.conv2d(ci, ky, padding=1)

        grad.append(grad_x + grad_y)
    grad = torch.concat(grad, dim=1)

    return grad



class LightingNet(nn.Module):
    '''
        define lighting network
    '''

    def __init__(self, in_channels, out_channels=27):
        super(LightingNet, self).__init__()
        mid_channels = 128


        self.pool = nn.AvgPool2d(16, stride=1, padding=0)
        self.down_fc1 = nn.Linear(in_channels, mid_channels)
        self.down_fc2 = nn.Linear(mid_channels, out_channels)

        self.up_fc1 = nn.Linear(out_channels, mid_channels)
        self.up_fc2 = nn.Linear(mid_channels, in_channels)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def down(self, feat):
        # light = self.conv(feat)
        light = self.pool(feat)
        light = light.view(feat.size(0), -1)
        light = self.down_fc1(light)
        light = self.down_fc2(light)
        return light

    def up(self, target_light):
        feat = self.up_fc1(target_light)
        feat = self.up_fc2(feat)
        feat = feat.view(target_light.size(0), -1, 1, 1)
        return feat


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=16 * 16, num_filter=16):
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filter = num_filter

        self.down_1 = conv_block_2(self.in_channels, self.num_filter)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter, self.num_filter * 2)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter * 2, self.num_filter * 4)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter * 4, self.num_filter * 8)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(self.num_filter * 8, out_channels)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        features = []
        features.append(down_1)
        features.append(down_2)
        features.append(down_3)
        features.append(down_4)

        return bridge, features


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_filter=16):
        super(UNetDecoder, self).__init__()

        self.in_channels = in_channels  # 256
        self.out_channels = out_channels  # 3
        self.num_filter = num_filter

        self.trans_4 = conv_trans_block(self.in_channels, self.num_filter * 8)
        self.up_4 = conv_block_2(self.num_filter * 16, self.num_filter * 8)
        self.trans_3 = conv_trans_block(self.num_filter * 8, self.num_filter * 4)
        self.up_3 = conv_block_2(self.num_filter * 8, self.num_filter * 4)
        self.trans_2 = conv_trans_block(self.num_filter * 4, self.num_filter * 2)
        self.up_2 = conv_block_2(self.num_filter * 4, self.num_filter * 2)
        self.trans_1 = conv_trans_block(self.num_filter * 2, self.num_filter)
        self.up_1 = conv_block_2(self.num_filter * 2, self.num_filter)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, out_channels, 3, 1, 1),
            nn.Tanh()
            # nn.Sigmoid()
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input, features):
        down_1, down_2, down_3, down_4 = features

        trans_4 = self.trans_4(input)
        concat_4 = torch.cat([trans_4, down_4], dim=1)
        up_4 = self.up_4(concat_4)
        trans_3 = self.trans_3(up_4)
        concat_3 = torch.cat([trans_3, down_3], dim=1)
        up_3 = self.up_3(concat_3)
        trans_2 = self.trans_2(up_3)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_1 = self.trans_1(up_2)
        concat_1 = torch.cat([trans_1, down_1], dim=1)
        up_1 = self.up_1(concat_1)

        out = self.out(up_1)

        return out
