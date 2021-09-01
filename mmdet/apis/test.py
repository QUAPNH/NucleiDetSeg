import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np
import mmcv
import matplotlib.pyplot as plt
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import torch
import cv2
import os
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist
from mmcv.runner import get_dist_info
from scipy.stats import pearsonr

from mmdet.core import encode_mask_results, tensor2imgs


def parse_rec(annotation):
    objects = []
    bboxes, labels = annotation
    for box in bboxes:
        obj_struct = {}
        obj_struct['name'] = 'cell'
        obj_struct['bbox'] = box
        objects.append(obj_struct)
    return objects


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def voc_ap(rec, prec, use_07_metric=True):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score

def get_fast_aji_plus(true, pred):
    """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


def iou_test(model,
             dsets,
             data_loader,
             img_height,
             img_width,
             show=False,
             out_dir=None,
             show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    ious = []
    scores = []
    plt.figure(figsize=(5, 5))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            img_metas = data['img_metas'][0].data[0]
            img_meta = img_metas[0]
            ori_h, ori_w = img_meta['ori_shape'][:-1]
            # print(img_metas)
            BBGT_box, BBGT_label, BBGT_mask, gtcode = dsets.load_annotation(dsets.img_files[i])
            # print(gtcode)
            result = model(return_loss=False, rescale=True, **data)
            for bbox_results, mask_results in result:
                mask_results = mask_results[0]
                bbox_results = bbox_results[0]
                l = len(BBGT_mask)
                nd = len(mask_results)
                # 计算与gt的iou最大的预测mask，并将其标记，并与gt-mask进行计算aji
                for d in range(len(bbox_results)):
                    d_BB_bbox = bbox_results[d]
                    # print(d_BB_bbox)
                    score = d_BB_bbox[4]
                    # print("score")
                    # print(score)
                    ovmax = -100
                    if score < 0.5:
                        continue
                    for g in range(len(BBGT_box)):
                        gt_bbox = BBGT_box[g]
                        # print(d_BB_bbox.shape)
                        if d_BB_bbox.shape[0] == 5:
                            d_BB_bbox = d_BB_bbox[:4]
                        overlaps = compute_iou(gt_bbox, d_BB_bbox)
                        if overlaps > ovmax:
                            ovmax = overlaps

                    if ovmax >= 0.5:
                    # print("ovmax")
                    # print(ovmax)
                        ious.append(ovmax)
                        scores.append(score)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                # 创建背景为纯黑色的图片，将分割结果画在上面(自己改动)
                # img_show = np.zeros([ori_w, ori_h, 3], np.uint8)
                # img_show = mmcv.imresize(img_show, (ori_w, ori_h))


                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    print(len(ious))
    print("--------------")
    # print(ious)
    # plt.title("")  # 显示图表名称
    pccs = pearsonr(ious, scores)
    print(pccs)
    plt.xlabel("IoU")  # x轴名称
    plt.ylabel("Calibrated Box Score")  # y轴名称
    plt.scatter(ious, scores, s=1)
    plt.savefig('./Calibrated Box Score VS  IoU.png')
    plt.show()


    return results


def aji_test(model,
             dsets,
             data_loader,
             img_height,
             img_width,
             show=False,
             out_dir=None,
             show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    ajis = []
    recs = {}
    all_scores = []
    # for i, img_file in enumerate(dsets.img_files):
    #     recs[img_file] = parse_rec(dsets.load_annotation(img_file))
    # nd = len(all_img_files)
    # tp = np.zeros(nd)
    # fp = np.zeros(nd)
    class_recs = {}
    npos = 0
    all_tp = []
    all_fp = []
    # for img_file in dsets.img_files:
    #     R = [obj for obj in recs[img_file] if obj['name'] == "cell"]
    #     bbox = np.array([x['bbox'] for x in R])
    #     det = [False] * len(R)
    #     npos = npos + len(R)
    #     class_recs[img_file] = {'bbox': bbox, 'det': det}

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            img_metas = data['img_metas'][0].data[0]
            img_meta = img_metas[0]
            ori_h, ori_w = img_meta['ori_shape'][:-1]
            # print(img_metas)
            BBGT_box, BBGT_label, BBGT_mask, gtcode = dsets.load_annotation(dsets.img_files[i])
            gt_masks = np.zeros([ori_h, ori_w], np.uint8)
            # print(gtcode)
            for mask in BBGT_mask:
                mask = np.array(mask).astype('uint8')
                gt_masks = cv2.bitwise_or(mask, gt_masks)
            result = model(return_loss=False, rescale=True, **data)
            for bbox_results, mask_results in result:
                mask_results = mask_results[0]
                bbox_results = bbox_results[0]
                BB_conf = bbox_results[:,4]
                sorted_ind = np.argsort(-BB_conf)
                BB_conf = BB_conf[sorted_ind]
                all_scores.extend(BB_conf)
                # print("++++++++++++")
                # print(sorted_ind)
                l = len(BBGT_mask)
                nd = len(mask_results)
                tp = np.zeros(nd)
                fp = np.zeros(nd)
                nd_gt = BBGT_box.shape[0]
                npos = npos + nd_gt
                # 计算与gt的iou最大的预测mask，并将其标记，并与gt-mask进行计算aji
                gt_flag = np.zeros((l))
                det_flag = np.zeros((nd))
                # print(len(mask_results))
                # print("ff")
                # print(len(bbox_results[0]))
                # print(len(BBGT_mask))
                # print(len(BBGT_box))
                for d in range(len(mask_results)):
                    d_BB_mask = mask_results[d]
                    # d_BB_box =
                    # print(d_BB_mask.shape)
                    # d_BB_bbox = bbox_results[0][d]
                    ovmax = -1000
                    gmax = -1
                    dmax = -1
                    for g in range(len(BBGT_mask)):
                        gt_mask = BBGT_mask[g]
                        # gt_bbox = BBGT_box[g]
                        # iou = bbox_overlaps(gt_bbox, d_BB_bbox, is_aligned=True)
                        overlaps = mask_iou(d_BB_mask, gt_mask)
                        if overlaps > ovmax:
                            ovmax = overlaps
                            gmax = g
                            dmax = d

                    if ovmax > 0.5:
                        if not gt_flag[gmax]:
                            tp[d] = 1.
                            gt_flag[gmax] = 1
                            det_flag[dmax] = 1
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.
                all_fp.extend(fp)
                all_tp.extend(tp)
                det_flag = np.nonzero(det_flag)[0]
                mask_res = []
                img_mask = np.zeros([ori_h, ori_w], np.uint8)
                for j in det_flag:
                    j = int(j)
                    d_mask = mask_results[j]
                    mask_res.append(d_mask)

                for mask in mask_res:
                    mask = np.array(mask).astype('uint8')
                    img_mask = cv2.bitwise_or(mask, img_mask)
            pred = remap_label(img_mask, by_size=False)
            true = remap_label(gt_masks, by_size=False)
            aji = get_fast_aji_plus(true, pred)
            print("aji:", aji)
            ajis.append(aji)


        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                # 创建背景为纯黑色的图片，将分割结果画在上面(自己改动)
                # img_show = np.zeros([ori_w, ori_h, 3], np.uint8)
                # img_show = mmcv.imresize(img_show, (ori_w, ori_h))


                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    all_fp = np.asarray(all_fp)
    all_tp = np.asarray(all_tp)
    all_scores = np.asarray(all_scores)
    sorted_ind = np.argsort(-all_scores)
    all_fp = all_fp[sorted_ind]
    all_tp = all_tp[sorted_ind]
    all_fp = np.cumsum(all_fp)
    all_tp = np.cumsum(all_tp)
    rec = all_tp / float(npos)
    prec = all_tp / np.maximum(all_tp + all_fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    print("ap@0.5的值为：" )
    print(ap)
    return results, ajis


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # all_scores = []
    # all_ious = []
    # plt.figure(figsize=(10, 10), dpi=100)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # all_scores.append(scores)
            # all_ious.append(ious)
            # print(scores)
            # print(ious)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                # 创建背景为纯黑色的图片，将分割结果画在上面(自己改动)
                img_show = np.zeros([ori_w, ori_h, 3], np.uint8)
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    # all_scores = torch.cat(all_scores)
    # all_scores = np.array(all_scores.cpu())
    # all_ious = torch.cat(all_ious)
    # all_ious = np.array(all_ious.cpu())
    # plt.scatter(all_scores, all_ious, s=0.01)
    # plt.show()

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
