import os
import argparse
from tqdm import tqdm
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from DISTS_pytorch import DISTS

import torch
import torch.nn as nn

from datasets.dataset import Dataset
from models.model import ShadowRemoval
from utils.visualize import Visualize
from utils.metrics import PSNR, SSIM
from utils.misc import AverageMeter
from utils.shading import sh2lighting


import lpips


def arg_parser():
    parser = argparse.ArgumentParser(description='shadow')
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--dataroot', default='./data/PSM/evaluation/input')
    parser.add_argument('--load_ckpt', type=str,
                        default='./checkpoints/best_model.pth')
    parser.add_argument('--batch_size', '-b', default=1, type=int)
    parser.add_argument('--num_workers', '-nw', default=8, type=int)
    parser.add_argument('--imsize', default=256, type=int)
    parser.add_argument('--use_visdom', action='store_true')
    parser.add_argument('--display', default=500, type=int)
    parser.add_argument('--save', type=str,
                        default=None
                        # default='./result/YaleB'
    )

    return parser.parse_args()

def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = arg_parser()

    fix_seed()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    print('Load dataset...')
    dataset = Dataset(args, train=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    if args.use_visdom:
        vis = Visualize(args)

    print('Load model checkpoint: ' + args.load_ckpt)
    trainer = ShadowRemoval(args, train_mode=False).cuda()
    trainer.load(args.load_ckpt)
    trainer.eval()

    if args.save is not None:
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    rmse = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    lpips_error = AverageMeter()

    dists = AverageMeter()
    D = DISTS().cuda()

    print('Start inference')

    trainer.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].float().cuda()
            img = data['input']

            if img.size(1) == 3:
                rgb = img.clone()
                yuv = rgb_to_yuv(rgb)
                y = yuv[:, 0, :, :].unsqueeze(1)
                u = yuv[:, 1, :, :].unsqueeze(1)
                v = yuv[:, 2, :, :].unsqueeze(1)

                y_out = trainer.predict(y)

                yuv_out = torch.cat([y_out, u, v], dim=1)
                output = yuv_to_rgb(yuv_out)
                output = torch.clamp(output, 0, 1)

                if args.use_visdom:
                    vis.show_images(y, win='y')
                    vis.show_images(y_out, win='y_output')

                    trainer.visualize(vis, {'input': y}, use_gt=False)

            else:
                output = trainer.predict(img)

                if args.use_visdom:
                    trainer.visualize(vis, data)

            if args.use_visdom:
                vis.show_images(img, win='input')
                vis.show_images(output, win='output')


            if args.save is not None:
                b, c, h, w = img.shape
                for i in range(b):
                    save_path = os.path.join(args.save, data['img_name'][i].split('/')[-1])
                    out_img = output[i][0].permute(0, 1).cpu().numpy()
                    out_img = (out_img * 255).astype(np.uint8)
                    out_img = Image.fromarray(out_img, 'L')
                    out_img.save(save_path)

            if 'gt' in data.keys():
                gt = data['gt'].float().to(device)
                n = output.size(0)

                rmse.update(torch.sqrt(nn.MSELoss()(output, gt)).float(), n=n)
                psnr.update(PSNR()(output, gt), n=n)
                s = SSIM()(output, gt)
                ssim.update(s, n=n)

                dists.update(D(output, gt, batch_average=True), n=n)

                d = loss_fn_alex(output, gt).mean()
                lpips_error.update(d, n=n)

    if ssim.count != 0:
        print('RMSE: {:.4f}, PSNR: {:.2f}, SSIM: {:.4f}'.format(rmse.avg, psnr.avg, ssim.avg))

        dssim = (1 - ssim.avg) / 2
        print('DSSIM: {:.4f}, LPIPS: {:.4f}'.format(dssim, lpips_error.avg))
        print('DISTS: {:.4f}'.format(dists.avg))



def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out

def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    g: torch.Tensor = y + -0.396 * u - 0.581 * v
    b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out

if __name__ == "__main__":
    main()
