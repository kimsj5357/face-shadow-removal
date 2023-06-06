import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from scipy.optimize import curve_fit
from skimage.transform import estimate_transform, warp
from skimage.color import rgb2yuv, yuv2rgb, rgb2gray
from skimage.transform import resize
from tqdm import tqdm
import time
import scipy.io as sio

from utils.shading import get_lighting, sh2lighting
from models.unet import UNet
from utils.render import render_shadow, get_shadow


class YaleBDataset():
    def __init__(self, opt, train=True, rgb=False):

        self.opt = opt
        self.dataroot = './data/ExtendedYaleB_png'
        self.imsize = opt.imsize
        self.train = train

        self.img_path = os.path.join(self.dataroot)
        list_file = 'train.txt' if self.train else 'test.txt'
        self.img_list = np.loadtxt(os.path.join(self.img_path, list_file), dtype=str)


        self.shadow_mask_path = './data/YaleB_3D/light'

        self.params_save_path = './data/params'

        self.mesh_path = './data/YaleB_3D/3D_3DDFA'

        self.face_crop_path = './data/cropface_DeepFace'

        self.face_render_path = './data/YaleB_3D/photometric'


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        data = self.get_imgs(img_name)

        return data

    def get_imgs(self, img_name):
        img_dir = img_name.split('/')[0]
        img_pose = img_name.split('_')[1][:3]
        front_name = os.path.join(img_dir, '{}_{}A+000E+00.png'.format(img_dir, img_pose))
        ambient_name = os.path.join(img_dir, '{}_{}_Ambient.png'.format(img_dir, img_pose))

        data = dict()

        img = Image.open(os.path.join(self.img_path, img_name))
        ori_img = np.array(img) / 255.
        img = img.resize((self.imsize, self.imsize))
        img = np.array(img) / 255.
        data['input'] = img[None, ...]

        front_img = Image.open(os.path.join(self.img_path, front_name))
        ori_front_img = np.array(front_img) / 255.
        front_img = front_img.resize((self.imsize, self.imsize))
        front_img = np.array(front_img) / 255.
        data['gt'] = front_img[None, ...]

        azimuth, elevation = int(img_name[-12:-8]), int(img_name[-7:-4])
        e = elevation * np.pi / 180.
        a = azimuth * np.pi / 180.
        x = np.cos(e) * np.sin(-a)
        y = np.sin(e)
        z = np.cos(e) * np.cos(-a)
        light = np.array([x, y, z])
        data['light'] = light

        # [1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-Y^2]
        sh = np.array([1, y, z, x, y*x, y*z, 3*z*z-1, x*z, x*x-y*y])
        data['sh'] = sh

        shadow_mask_path = os.path.join(self.shadow_mask_path, img_name.split('/')[1])
        shadow_mask = Image.open(shadow_mask_path).convert('L')
        shadow_mask = shadow_mask.resize((self.imsize, self.imsize))
        shadow_mask = np.array(shadow_mask) / 255.
        data['shadow_mask'] = shadow_mask[None, ...]

        face_mask_path = os.path.join(self.shadow_mask_path, front_name.split('/')[1])
        face_mask = Image.open(face_mask_path).convert('L')
        face_mask = face_mask.resize((self.imsize, self.imsize))
        face_mask = np.array(face_mask) / 255.
        face_mask = (face_mask > 0) * 1.
        data['face_mask'] = face_mask[None, ...]


        gt_name = front_name.split('/')[-1].split('.')[0]
        mesh = sio.loadmat(os.path.join(self.face_render_path, gt_name, gt_name + '.mat'))

        normals = mesh['normal_image'][0].transpose(1, 2, 0)
        normals = resize(normals, (self.imsize, self.imsize))
        data['normal'] = normals.transpose(2, 0, 1)

        albedo = mesh['albedo_image'].squeeze() / 255.
        albedo = rgb2gray(albedo)
        albedo = resize(albedo, (self.imsize, self.imsize))
        data['albedo'] = np.expand_dims(albedo, 0)


        shading = get_shading(normals, light)
        gt_light = np.array([0, 0, 1])
        gt_shading = get_shading(normals, gt_light)
        shadow_part_mask = gt_shading - shading
        shadow_part_mask = np.clip(shadow_part_mask, 0, 1)
        data['shadow_part_mask'] = np.expand_dims(shadow_part_mask, 0)

        face_mask = (normals[:, :, 0] != 0).astype(np.float32)
        data['face_mask'] = np.expand_dims(face_mask, 0)

        highlight_part_mask = shading - gt_shading
        highlight_part_mask = np.clip(highlight_part_mask, 0, 1)
        highlight_part_mask *= face_mask
        data['highlight_part_mask'] = np.expand_dims(highlight_part_mask, 0)




        normal = np.reshape(normals, (-1, 3))

        # sh = get_SH(phi, theta)
        sh = np.array([1, y, z, x, y * x, y * z, 3 * z * z - 1, x * z, x * x - y * y]) * 0.7
        data['light_direction'] = np.array([a, e])

        data['ori_img'] = ori_img
        data['img_name'] = img_name

        return data


def get_shading(N, L):
    c1 = 0.8862269254527579
    c2 = 1.0233267079464883
    c3 = 0.24770795610037571
    c4 = 0.8580855308097834
    c5 = 0.4290427654048917

    nx = N[:, :, 0]
    ny = N[:, :, 1]
    nz = N[:, :, 2]

    h, w, c = N.shape

    Y1 = c1 * np.ones((h, w))
    Y2 = c2 * ny
    Y3 = c2 * nz
    Y4 = c2 * nx
    Y5 = c4 * ny * nx
    Y6 = c4 * ny * nz
    Y7 = c3 * (3 * nz * nz - 1)
    Y8 = c4 * nx * nz
    Y9 = c5 * (nx * nx - ny * ny)

    x, y, z = L
    sh = np.array([1, y, z, x, y * x, y * z, 3 * z * z - 1, x * z, x * x - y * y])


    sh = np.tile(sh, h * w).reshape(h, w, 9)
    shading = Y1 * sh[:, :, 0] + Y2 * sh[:, :, 1] + Y3 * sh[:, :, 2] + \
              Y4 * sh[:, :, 3] + Y5 * sh[:, :, 4] + Y6 * sh[:, :, 5] + \
              Y7 * sh[:, :, 6] + Y8 * sh[:, :, 7] + Y9 * sh[:, :, 8]

    return shading


def get_SH(phi, theta):
    sh = np.zeros(9, dtype=np.float32)
    sh[0] = np.sqrt(1/4/np.pi)
    sh[1] = sh[0] * np.sqrt(3) * np.cos(phi) * np.sin(theta)
    sh[2] = sh[0] * np.sqrt(3) * np.cos(theta)
    sh[3] = sh[0] * np.sqrt(3) * np.sin(phi) * np.sin(theta)
    sh[4] = sh[0] * np.sqrt(15) * np.sin(phi) * np.cos(phi) * (np.sin(theta) ** 2)
    sh[5] = sh[0] * np.sqrt(15) * np.sin(phi) * np.sin(theta) * np.cos(theta)
    sh[6] = sh[0] * np.sqrt(5/4) * (3 * np.cos(theta) ** 2 - 1)
    sh[7] = sh[0] * np.sqrt(15) * np.cos(phi) * np.sin(theta) * np.cos(theta)
    sh[8] = sh[0] * np.sqrt(15/4) * (np.cos(phi) ** 2 - np.sin(phi) ** 2) * (np.sin(theta) ** 2)
    return sh


def get_shading_SH(N, SH):
    """
    :param N: (N, 3)
    :param SH: (9, )
    :return: (N, )
    """
    norm_X = N[:, 0]
    norm_Y = N[:, 1]
    norm_Z = N[:, 2]

    numElem = N.shape[0]

    sh_basis = np.zeros((numElem, 9))
    att = np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])
    sh_basis[:, 0] = 0.5 / np.sqrt(np.pi) * att[0]

    sh_basis[:, 1] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Y * att[1]
    sh_basis[:, 2] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Z * att[1]
    sh_basis[:, 3] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_X * att[1]

    sh_basis[:, 4] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_X * att[2]
    sh_basis[:, 5] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_Z * att[2]
    sh_basis[:, 6] = np.sqrt(5) / 4 / np.sqrt(np.pi) * (3 * norm_Z ** 2 - 1) * att[2]
    sh_basis[:, 7] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_X * norm_Z * att[2]
    sh_basis[:, 8] = np.sqrt(15) / 4 / np.sqrt(np.pi) * (norm_X ** 2 - norm_Y ** 2) * att[2]

    shading = np.matmul(sh_basis, SH)

    return shading


def pad_img(img, imsize=256):
    ori_h, ori_w = img.shape[:2]
    ratio = imsize / max(ori_h, ori_w)
    h, w = int(ori_h * ratio), int(ori_w * ratio)

    resize_img = resize(img, (h, w))

    if img.ndim == 2:
        padded_img = np.zeros((imsize, imsize))
        padded_img[(imsize - h) // 2:(imsize - h) // 2 + h, (imsize - w) // 2:(imsize - w) // 2 + w] = resize_img

    else:  # img.ndim == 3
        padded_img = np.zeros((imsize, imsize, 3))
        padded_img[(imsize - h) // 2:(imsize - h) // 2 + h, (imsize - w) // 2:(imsize - w) // 2 + w, :] = resize_img

    return padded_img, ratio

def lit(x, a, b):
    return a * x + b
def gamma(x, r):
    return np.exp(r * np.log(x))
def Schlick_bias(x, a):
    return x / ((1 / a - 2) * (1 - x) + 1)

