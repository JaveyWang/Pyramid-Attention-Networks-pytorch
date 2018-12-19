import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

def save_model(state, directory='./checkpoints', filename=None):
    if os.path.isdir(directory):
        pkl_filename = os.path.join(directory, filename)
        torch.save(state, pkl_filename)
        print('Save "{:}" in {:} successful'.format(pkl_filename, directory))
    else:
        print(' "{:}" directory is not exsits!'.format(directory))

def onehot_mask(mask, num_cls=21):
    """
    :param mask: label of a image. tensor shape:[h, w]
    :param num_cls: number of class. int
    :return: onehot encoding mask. tensor shape:[num_cls, h, w]
    """
    b, h, w = mask.shape
    mask_onehot = torch.zeros_like(mask).unsqueeze(1).expand(b, num_cls, h, w)
    mask_onehot = mask_onehot.scatter_(1, mask.long().unsqueeze(1), 1)
    return mask_onehot

def get_each_cls_iu(pred, gt):
    r"""
    This function is used for getting mean Intersaction over Union (mIOU).
    :param pred: numpy array: [B, H, W]. The prediction of mask.
    :param gt: numpy array: [B, H, W]. The ground truth mask.
    :return: i_count, u_count. numpy array: [21]
    """
    assert (pred.shape == gt.shape), "pred shape: {:}, ground truth shape: {:}".format(pred.shape, gt.shape)
    gt = gt.astype(np.int32)
    pred = pred.astype(np.int32)

    max_label = 20  # labels from 0,1, ... 20(for VOC)
    i_count = np.zeros((max_label + 1,))
    u_count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        i_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        i_count[j] = float(len(i_jj))
        u_count[j] = float(len(u_jj))

    return i_count, u_count

def set_parameters_grad(networks, isgrad=True):
    for n in networks:
        for p in n.parameters():
            if isgrad:
                p.requires_grad=True
            else:
                p.requires_grad=False

class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch/self.max_iter) ** self.power
                for base_lr in self.base_lrs]