# -*- coding:utf-8 -*-
from __future__ import division
# 在python2 中导入未来的支持的语言特征中division(精确除法)

import numpy as np
import six  # solve the compatibility problems between python2 and python3

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """collect a confusion matrix.

    The number of classes: n\_class is 'max(pred\ labels, gt\_labels) + 1',
    which is the maximum class id of the inputs added by one

    Args
        pred_labels (iterable of numpy.ndarry):A collection of perdicted labels.
        the shape of a albel array is (H,W), H: label height, W :label width

        gt_labels (iterable of numpy.ndarry):A collection of ground
        truth labels. The shape of a ground truth label array is
        (H,W), and its corresponding prediction label should have
        the same shape.
        A pixel with value '-1' will be ignored during evaluation.

    Returns:
         numpy.ndarry
         A confusion matrix. its shape is (n\_class, n\class)
         the (i,j) th element corresponds to the number of pixels
         that are labeled as class 'i' by the ground truth and
         class 'j' by the prediction.
    """
    pred_labels = iter(pred_labels)

    gt_labels = iter(gt_labels)

    n_class = 32
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be 2.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0: n_class, 0: n_class] = confusion
            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class ** 2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # this code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('length of input iterables need to be same.')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    """calculate intersection over union with a given confusion matrix"""
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou[:-1]


def eval_semantic_segmentation(pred_labels, gt_labels):
    """Evaluate metrics used in semantic segmentation.
    includes: IoU, pixel accuracy and class accuracy
    for the task of semantic segmentation."""
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum()/confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)

    return {'iou': iou,
            'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy[:-1])
            }






