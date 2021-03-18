from __future__ import print_function
from . import model_dict
import sys

import numpy as np

import torch
from torch.nn import functional as F


def create_model(name, n_cls, dataset='miniImageNet'):
    """create model by name"""
    print("AAA")
    if dataset == 'miniImageNet' or dataset == 'tieredImageNet':
        if name.endswith('v2') or name.endswith('v3'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet50'):
            print('use imagenet-style resnet50')
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls)
        elif name.startswith('wrn'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    elif dataset == 'CIFAR-FS' or dataset == 'FC100':
        if name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))
    print("AAA")
    print(model)
    return model


def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]


def conv_orth_dist(kernel, stride=1):
    [o_c, i_c, w, h] = kernel.shape
    assert (w == h), "Do not support rectangular kernel"
    # half = np.floor(w/2)
    assert stride < w, "Please use matrix orthgonality instead"
    new_s = stride * (w - 1) + w  # np.int(2*(half+np.floor(half/stride))+1)
    temp = torch.eye(new_s * new_s * i_c).reshape((new_s * new_s * i_c, i_c, new_s, new_s)).cuda()
    out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s * new_s * i_c, -1))
    Vmat = out[np.floor(new_s ** 2 / 2).astype(int)::new_s ** 2, :]
    temp = np.zeros((i_c, i_c * new_s ** 2))
    for i in range(temp.shape[0]): temp[i, np.floor(new_s ** 2 / 2).astype(int) + new_s ** 2 * i] = 1
    return torch.norm(Vmat @ torch.t(out) - torch.from_numpy(temp).float().cuda())


def deconv_orth_dist(kernel, stride=2, padding=1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1] / 2))
    target[:, :, ct, ct] = torch.eye(o_c).cuda()
    return torch.norm(output - target)


def orth_dist(mat, stride=None):
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1, 0)
    return torch.norm(torch.t(mat) @ mat - torch.eye(mat.shape[1]).cuda())

