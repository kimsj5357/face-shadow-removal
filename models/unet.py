import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_trans_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


def conv_block_2(in_channels, out_channels):
    return nn.Sequential(
        conv_block(in_channels, out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels)
    )


class UNet(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_filter = 16

        self.down_1 = conv_block_2(in_dims, self.num_filter)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter, self.num_filter * 2)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter * 2, self.num_filter * 4)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter * 4, self.num_filter * 8)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(self.num_filter * 8, self.num_filter * 16)
        # self.bridge = conv_block_2(self.num_filter * 2, self.num_filter * 4)

        self.trans_4 = conv_trans_block(self.num_filter * 16, self.num_filter * 8)
        self.up_4 = conv_block_2(self.num_filter * 16, self.num_filter * 8)
        self.trans_3 = conv_trans_block(self.num_filter * 8, self.num_filter * 4)
        self.up_3 = conv_block_2(self.num_filter * 8, self.num_filter * 4)
        self.trans_2 = conv_trans_block(self.num_filter * 4, self.num_filter * 2)
        self.up_2 = conv_block_2(self.num_filter * 4, self.num_filter * 2)
        self.trans_1 = conv_trans_block(self.num_filter * 2, self.num_filter)
        self.up_1 = conv_block_2(self.num_filter * 2, self.num_filter)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, out_dims, 3, 1, 1),
            nn.Tanh()
        )

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
        self.feat = bridge
        # bridge = self.bridge(pool_2)
        # trans_2 = self.trans_2(bridge)

        trans_4 = self.trans_4(bridge)
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

    def loss(self, pred, gt):
        criterion = nn.MSELoss()
        loss = criterion(pred, gt)
        return loss

    def get_optimizer_params(self, lr, weight_decay):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        return params

