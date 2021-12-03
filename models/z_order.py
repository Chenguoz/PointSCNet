# import numpy
import numpy as np
import torch

from matplotlib import pyplot as plt
import time


def round_to_int_32(data):
    """
    Takes a Numpy array of float values between
    -1 and 1, and rounds them to significant
    32-bit integer values, to be used in the
    morton code computation

    :param data: multidimensional numpy array
    :return: same as data but in 32-bit int format
    """
    # first we rescale points to 0-512
    data = 256 * (data + 1)
    # now convert to int
    data = np.round(2 ** 21 - data).astype(dtype=np.int32)

    return data


def split_by_3(x):
    """
    Method to separate bits of a 32-bit integer
    by 3 positions apart, using the magic bits
    https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/

    :param x: 32-bit integer
    :return: x with bits separated
    """
    # we only look at 21 bits, since we want to generate
    # a 64-bit code eventually (3 x 21 bits = 63 bits, which
    # is the maximum we can fit in a 64-bit code)
    x &= 0x1fffff  # only take first 21 bits
    # shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | (x << 32)) & 0x1f00000000ffff
    # shift left 16 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    # shift left 8 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    # shift left 4 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    # shift left 2 bits, OR with self, and 0001001001001001001001001001001001001001001001001001001001001001
    x = (x | (x << 2)) & 0x1249249249249249

    return x


def get_z_order(x, y, z):
    """
    Given 3 arrays of corresponding x, y, z
    coordinates, compute the morton (or z) code for
    each point and return an index array
    We compute the Morton order as follows:
        1- Split all coordinates by 3 (add 2 zeros between bits)
        2- Shift bits left by 1 for y and 2 for z
        3- Interleave x, shifted y, and shifted z
    The mordon order is the final interleaved bit sequence

    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :return: index array with morton code
    """
    res = 0
    res |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2

    return res


def get_z_values(data):
    """
    Computes the z values for a point array
    :param data: Nx3 array of x, y, and z location

    :return: Nx1 array of z values
    """
    data = data.cpu().detach().numpy()
    points_round = round_to_int_32(data)  # convert to int
    z = get_z_order(points_round[:, 0], points_round[:, 1], points_round[:, 2])

    return z


def pointnet_index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # print(view_shape)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # print(repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # print(batch_indices)
    # print(batch_indices.shape, idx.shape)
    new_points = points[batch_indices, idx, :]
    # print(new_points)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def z_order_point_sample(xyz, npoints):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape

    if npoints >= N:
        return torch.linspace(0, N, steps=N, dtype=int).view(1, N).repeat(B, 1)

    centroids = torch.zeros(B, npoints, dtype=int).to(device)
    for batch_idx in range(B):
        z = get_z_values(xyz[batch_idx])
        z = np.argsort(z)
        centroids[batch_idx, :] = torch.from_numpy(
            z[torch.linspace(0, N - 1, steps=npoints, dtype=int)].reshape(1, npoints))

    return centroids
