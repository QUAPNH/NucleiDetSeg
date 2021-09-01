import torch
from mmcv.ops.nms import batched_nms
import numpy as np
import matplotlib.pyplot as plt
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import math

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]
    # print("fusion-socres")
    # print(scores)

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]
    # print("lables")
    # print(valid_mask)
    # print("bbox")
    # print(len(bboxes))

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    # print("bbox2")
    # print(len(dets))
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    # print("bbox3")
    # print(len(dets))

    return dets, labels[keep]


def multiclass_nms_score(multi_bboxes,
                         multi_cls_scores,
                         multi_iou_scores,
                         score_thr,
                         nms_cfg,
                         max_num=-1,
                         score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_cls_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_cls_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_cls_scores.size(0), num_classes, 4)
    cls_scores = multi_cls_scores[:, :-1]
    # print(cls_scores)
    iou_scores = multi_iou_scores
    # print("cls_scores")
    # print(cls_scores)
    # print("iou_scores")
    # print(iou_scores)
    nms_scores = (cls_scores * iou_scores).sqrt()
    # nms_scores = (cls_scores * iou_scores)**(0.1)
    # nms_scores = math.pow(nms_scores, 1/3)
    # print(nms_scores)


    # filter out boxes with low scores
    valid_mask = nms_scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        cls_scores = cls_scores * score_factors[:, None]
        iou_scores = iou_scores * score_factors[:, None]
    cls_scores = torch.masked_select(cls_scores, valid_mask)
    iou_scores = torch.masked_select(iou_scores, valid_mask)
    nms_scores = torch.masked_select(nms_scores, valid_mask)

    # print("iou_scores")
    # print(iou_scores)
    # nms_scores = (cls_scores + iou_scores) / 2
    # nms_scores = (cls_scores*iou_scores).sqrt()
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels
    # scores = []
    # ious = []
    # plt.figure(figsize=(10, 10), dpi=100)
    # ovthresh = 0.49
    # for j in range(50):
    #     ovthresh += 0.01
    #     nms_cfg = {'type':'nms', 'iou_threshold':ovthresh}
        # print(nms_cfg)
    dets, keep = batched_nms(bboxes, nms_scores, labels, nms_cfg)
    # print("dets")
    # print(dets)
    # print("keep")
    # print(keep)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    # inds = dets[:,4] >= 0.5
    # score = dets[inds, 4]
        # score = np.array(score.cpu())
        # print(score)
        # print(score.shape)
        # len = score.shape[0]
        # iou = torch.tensor([ovthresh] * len)
        # scores.append(score)
        # ious.append(iou)
    # scores = torch.cat(scores)
    # scores = np.array(scores.cpu())
    # ious = torch.cat(ious)
    # ious = np.array(ious.cpu())
    # print(scores.shape)
    # print(ious.shape)
    # plt.scatter(scores, ious, s=1)
    # plt.show()
    # print("nms")
    # print(nms_scores)

    return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
