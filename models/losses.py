# Based on code from
# https://github.com/mkisantal/backboned-unet/blob/master/backboned_unet/utils.py

import torch
import torch.nn as nn
from torch.nn import functional as F


def dice_score(input, target, classes, ignore_index=-100):
    """ Functional dice score calculation. """
    target = target.long().unsqueeze(1)

    # getting mask for valid pixels,
    # then converting "void class" to background
    valid = target != ignore_index
    target[target == ignore_index] = 0
    valid = valid.float()

    # converting to onehot image with class channels
    onehot_target = torch.LongTensor(target.shape[0], classes, target.shape[-2], target.shape[-1]).zero_().cuda()
    onehot_target.scatter_(1, target, 1)  # write ones along "channel" dimension
    # classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0
    onehot_target = onehot_target.float()

    # keeping the valid pixels only
    onehot_target = onehot_target * valid
    input = input * valid

    dice = 2 * (input * onehot_target).sum([2, 3]) / ((input**2).sum([2, 3]) + (onehot_target**2).sum([2, 3]))
    return dice.mean(dim=1)


class DiceLoss(nn.Module):
    """ Dice score implemented as a nn.Module. """
    def __init__(self,
                 classes,
                 loss_mode='negative_log',
                 ignore_index=-1,
                 activation=None):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.loss_mode = loss_mode
        self.activation = activation

    def forward(self, input, target):
        if self.activation is not None:
            input = self.activation(input)

        score = dice_score(input, target, self.classes, self.ignore_index)
        if self.loss_mode == 'negative_log':
            eps = 1e-12
            return (-(score+eps).log()).mean()
        elif self.loss_mode == 'one_minus':
            return (1 - score).mean()
        else:
            raise ValueError('Loss mode unknown. Please use \'negative_log\' or \'one_minus\'!')