import torch.nn as nn
import torch.nn.functional as F
from utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from z_order import *


def get_relation_zorder_sample(input_u, input_v, random_sample=True, sample_size=64):
    batchsize, in_uchannels, length = input_u.shape
    _, in_vchannels, _ = input_v.shape
    device = input_u.device
    if not random_sample:
        sample_size = length

    input_u = input_u.permute(0, 2, 1)
    input_v = input_v.permute(0, 2, 1)
    ides = z_order_point_sample(input_u[:, :, :3], sample_size)

    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device).view(batchsize, 1).repeat(1, sample_size)

    temp_relationu = input_u[batch_indices, ides, :].permute(0, 2, 1)
    temp_relationv = input_v[batch_indices, ides, :].permute(0, 2, 1)
    input_u = input_u.permute(0, 2, 1)
    input_v = input_v.permute(0, 2, 1)
    relation_u = torch.cat([input_u.view(batchsize, -1, length, 1).repeat(1, 1, 1, sample_size),
                            temp_relationu.view(batchsize, -1, 1, sample_size).repeat(1, 1, length, 1)], dim=1)
    relation_v = torch.cat([input_v.view(batchsize, -1, length, 1).repeat(1, 1, 1, sample_size),
                            temp_relationv.view(batchsize, -1, 1, sample_size).repeat(1, 1, length, 1)], dim=1)

    return relation_u, relation_v, temp_relationu, temp_relationv



class PSCNChannelAttention(nn.Module):
    def __init__(self, channel_in):
        super(PSCNChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局自适应池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Conv1d(channel_in, channel_in // 2, bias=False, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_in // 2, channel_in, bias=False, kernel_size=1),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(channel_in, channel_in // 2, bias=False, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_in // 2, channel_in, bias=False, kernel_size=1),

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel_num, _ = x.size()

        avg_out = self.avg_pool(x).view(batch_size, channel_num, 1)  # squeeze操作
        max_out = self.max_pool(x).view(batch_size, channel_num, 1)

        avg_y = self.fc1(avg_out).view(batch_size, channel_num, 1)  # FC获取通道注意力权重，是具有全局信息的
        max_y = self.fc2(max_out).view(batch_size, channel_num, 1)
        out = self.sigmoid(avg_y + max_y)
        return out  # 注意力作用每一个通道上

        # return x + x * out.expand_as(x)  # 注意力作用每一个通道上


class PSCNSpatialAttention(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(PSCNSpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局自适应池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Sequential(
            nn.Conv1d(channel_in, channel_in // 2, bias=False, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_in // 2, channel_out, bias=False, kernel_size=1),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(channel_in, channel_in // 2, bias=False, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel_in // 2, channel_out, bias=False, kernel_size=1),

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel_num, point_num = x.size()

        avg_out = self.avg_pool(x).view(batch_size, channel_num, 1)  # squeeze操作
        max_out = self.max_pool(x).view(batch_size, channel_num, 1)

        avg_y = self.fc1(avg_out).view(batch_size, 1, point_num)  # FC获取通道注意力权重，是具有全局信息的
        max_y = self.fc2(max_out).view(batch_size, 1, point_num)
        out = self.sigmoid(avg_y + max_y)
        return out  # 注意力作用每一个空间上
        # return x + x * out.expand_as(x) # 注意力作用每一个空间上
        # return x + x * out.expand_as(x), out # 注意力作用每一个空间上


class PointSCN(nn.Module):
    def __init__(self, in_uchannels, in_vchannels, random_sample=True, sample_size=64):
        super(PointSCN, self).__init__()

        self.random_sample = random_sample
        self.sample_size = sample_size

        self.conv_gu = nn.Conv2d(2 * in_uchannels, 2 * in_uchannels, 1)
        self.bn1 = nn.BatchNorm2d(2 * in_uchannels)

        self.conv_gv = nn.Conv2d(2 * in_vchannels, 2 * in_vchannels, 1)
        self.bn2 = nn.BatchNorm2d(2 * in_vchannels)

        self.conv_uv = nn.Conv2d(2 * in_uchannels + 2 * in_vchannels, in_vchannels, 1)
        self.bn3 = nn.BatchNorm2d(in_vchannels)

        self.conv_f = nn.Conv1d(in_vchannels, in_vchannels, 1)
        self.bn4 = nn.BatchNorm1d(in_vchannels)

    def forward(self, input_u, input_v):
        """
              Input:
                  input_u: input points position data, [B, C, N]
                  input_v: input points data, [B, D, N]
              Return:
                  new_xyz: sampled points position data, [B, C, S]
                  new_points_concat: sample points feature data, [B, D', S]
        """

        relation_u, relation_v, _, _ = get_relation_zorder_sample(input_u, input_v, random_sample=self.random_sample,
                                                                  sample_size=self.sample_size)

        relation_uv = torch.cat(
            [F.relu(self.bn1(self.conv_gu(relation_u))), F.relu(self.bn2(self.conv_gv(relation_v)))], dim=1)

        relation_uv = F.relu(self.bn3(self.conv_uv(relation_uv)))

        relation_uv = torch.max(relation_uv, 3)[0]

        relation_uv = F.relu(self.bn4(self.conv_f(relation_uv)))
        relation_uv = torch.cat([input_v + relation_uv, input_u], dim=1)

        return relation_uv


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.4], [16, 128], in_channel,
                                             [[32, 32, 64], [64, 96, 128]])

        self.PSCN1 = PointSCN(3, 64 + 128, random_sample=True)
        self.attention1 = PSCNChannelAttention(128 + 64 + 3)
        self.attention3 = PSCNSpatialAttention(128 + 64 + 3, 256)

        self.sa2 = PointNetSetAbstraction(None, None, None, 128 + 64 + 3 + 3, [256, 512, 1024], True)  # 含池化得到

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l1_points = self.PSCN1(l1_xyz, l1_points)
        l1_points_att1 = self.attention1(l1_points)
        l1_points_att3 = self.attention3(l1_points)
        l1_points = l1_points * l1_points_att1.expand_as(l1_points) * l1_points_att3.expand_as(l1_points)
        l3_xyz, l3_points = self.sa2(l1_xyz, l1_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
