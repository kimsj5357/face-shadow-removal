import os
import argparse
from tqdm import tqdm
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

from datasets.dataset import Dataset
from models.model import ShadowRemoval
from utils.visualize import Visualize
from utils.metrics import PSNR, SSIM
from utils.misc import AverageMeter
from utils.shading import sh2lighting

from utils.metrics import SSIM, VGGPerceptualLoss, GradientLoss

import warnings
warnings.filterwarnings("ignore")

def arg_parser():
    parser = argparse.ArgumentParser(description='shadow')
    parser.add_argument('--dataset', default='YaleB')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--checkpoints_dir', default='./checkpoints')
    parser.add_argument('--batch_size', '-b', default=2, type=int)
    parser.add_argument('--num_workers', '-nw', default=1, type=int)
    parser.add_argument('--imsize', default=256, type=int)
    parser.add_argument('--use_visdom', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_ckpt', default='./checkpoints/best_model.pth')
    parser.add_argument('--display', default=100, type=int)

    parser.add_argument('--epoch', default=15, type=int)
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)

    parser.add_argument('--distributed', '-d', action='store_true')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)


    return parser.parse_args()

def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

def main():

    fix_seed()

    args = arg_parser()

    print_options(args)

    if args.checkpoints_dir is not None:
        args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.dataset)
    else:
        args.checkpoints_dir = os.path.join('./checkpoints', args.dataset)

    args.checkpoints_dir += '_' + args.model
    if args.use_D:
        args.checkpoints_dir += '_D'
    args.checkpoints_dir += args.ckpt_name
    print('Save checkpoint in ' + args.checkpoints_dir)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ngpus_per_node = torch.cuda.device_count()
    print('GPU: ', args.gpu)

    args.world_size = args.world_size * ngpus_per_node

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(int(args.gpu), ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    torch.cuda.set_device(gpu)

    trainer = ShadowRemoval(args)
    trainer.cuda(gpu)

    if args.rank % ngpus_per_node == 0:
        print_networks(trainer)

    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        args.batch_size = int(args.batch_size / ngpus_per_node)


        trainer = torch.nn.parallel.DistributedDataParallel(trainer,
                                                            device_ids=[gpu],
                                                            find_unused_parameters=True)


    print('Load dataset...')
    train_dataset = Dataset(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False if args.distributed else True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   sampler=train_sampler)

    test_dataset = Dataset(args, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)
    print('=> train images:', len(train_dataset))
    print('=> val images:', len(test_dataset))

    if args.use_visdom:
        print('Use visdom')
        vis = Visualize(args)


    lr = args.lr
    best_ssim = 0
    best_epoch = 0

    start_epoch = 0
    if args.resume:
        print('Load model checkpoint ' + args.load_ckpt)
        start_epoch = trainer.load(args.load_ckpt)

    eta = AverageMeter()
    total_iter = (args.epoch - start_epoch) * len(train_dataloader)

    print('Start training')
    for epoch in range(start_epoch, args.epoch):
        st = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        trainer.train()

        for i, data in enumerate(train_dataloader):

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].float().cuda()

            losses = trainer(data)

            t = time.time() - st
            eta.update(t)
            t = eta.avg * (total_iter - eta.count)

            if i % args.display == 0 and args.rank % ngpus_per_node == 0:
                log = 'Epoch: [{}/{}][{}/{}]\t LR: {:.5f}\t Loss: ['.format(
                    epoch, args.epoch, i, len(train_dataloader), lr)
                for key, value in losses.items():
                    log += '{}: {:.4f}, '.format(key, value.item())
                log = log[:-2] + ']\t ETA: {}'.format(time.strftime('%H:%M:%S', time.gmtime(t)))
                print(log)

                if args.use_visdom:
                    trainer.eval()
                    if isinstance(trainer, nn.parallel.DistributedDataParallel):
                        trainer.module.visualize(vis, data)
                    else:
                        trainer.visualize(vis, data)
                    trainer.train()

            st = time.time()


        if isinstance(trainer, nn.parallel.DistributedDataParallel):
            lr = trainer.module.adjust_learning_rate()
        else:
            lr = trainer.adjust_learning_rate()


        if args.rank % ngpus_per_node == 0:
            if isinstance(trainer, nn.parallel.DistributedDataParallel):
                trainer.module.save(epoch)
            else:
                trainer.save(epoch)

            psnr = AverageMeter()
            ssim = AverageMeter()
            trainer.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
                    img = data['input'].float().cuda()
                    gt = data['gt'].float().cuda()
                    if isinstance(trainer, nn.parallel.DistributedDataParallel):
                        output = trainer.module.predict(img)
                    else:
                        output = trainer.predict(img)
                    psnr.update(PSNR()(output, gt))
                    ssim.update(SSIM()(output, gt))

            if best_ssim < ssim.avg:
                if ssim.avg == np.inf:
                    continue

                best_ssim = ssim.avg.item()
                best_epoch = epoch

                shutil.copyfile(os.path.join(args.checkpoints_dir, 'epoch_%s.pth' % epoch),
                                os.path.join(args.checkpoints_dir, 'best_model.pth'))


            print('PSNR: {:.4f}, SSIM: {:.4f} (Best SSIM: {:.4f} at {} epoch)'.format(
                psnr.avg, ssim.avg, best_ssim, best_epoch))

    if args.rank % ngpus_per_node == 0:
        print('Best SSIM: {:.4f}, epoch: {}'.format(best_ssim, best_epoch))

    return best_ssim, best_epoch



def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

# print network information
def print_networks(model):
    print('---------- Networks initialized -------------')
    num_params = 0
    for name, layer in model.named_children():
        params = 0
        for param in layer.parameters():
            params += param.numel()
        if params > 0 and 'loss' not in name:
            print('{}: {:.3f} M'.format(name, params / 1e6))
            num_params += params
    # for param in model.parameters():
    #     num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')



if __name__ == "__main__":
    main()
