import torch
import numpy as np
from mmdet.core import bbox2result, bbox2roi, build_bbox_coder, multi_apply, multiclass_nms, multiclass_nms_score
from ..builder import HEADS, build_head, build_roi_extractor
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from .base_roi_head import BaseRoIHead
from .test_mixins_parallel import BBoxTestMixinParallel, MaskTestMixin
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class BBoxScoringRoIHead(StandardRoIHead, BBoxTestMixinParallel, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_iou_head,
                 num_classes=1,
                 bbox_coder=dict(
                   type='DeltaXYWHBBoxCoder',
                   target_means=[0., 0., 0., 0.],
                   target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_decoded_bbox=True,
                 **kwargs):
        assert bbox_iou_head is not None
        super(BBoxScoringRoIHead, self).__init__(**kwargs)
        self.bbox_iou_head = build_head(bbox_iou_head)
        self.num_classes = num_classes
        self.reg_decoded_bbox = reg_decoded_bbox
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.iou_history = []


    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(BBoxScoringRoIHead, self).init_weights(pretrained)
        self.bbox_iou_head.init_weights()


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """Forward function for training.

        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cur_iou = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # record the `iou_topk`-th largest IoU in an image
                # print("max_overlaps")
                # print(len(sampling_result.pos_inds))
                # iou_topk = len(sampling_result.pos_inds)
                # iou_topk = min(iou_topk,
                #                len(assign_result.max_overlaps))
                # if iou_topk == 0:
                #     cur_iou.append(iou_topk)
                # else:
                #     ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                #     cur_iou.append(ious[-1].item())
                sampling_results.append(sampling_result)
            # average the current IoUs over images
            # cur_iou = np.mean(cur_iou)
            # self.iou_history.append(cur_iou)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        # update IoU threshold and SmoothL1 beta
        # update_iter_interval = 400
        # if len(self.iou_history) % update_iter_interval == 0:
        #     new_iou_thr = self.update_hyperparameters()

        return losses


    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_iou, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        iou_targets = pos_bboxes.new_zeros((num_samples,))
        iou_labels = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        bbox_targets_maxone = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            # print(pos_gt_labels)
            labels[:num_pos] = pos_gt_labels
            # print(labels)
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            # print(pos_iou)
            iou_targets = pos_iou
            iou_labels[:num_pos] = 1


            pos_bbox_targets_maxone = pos_gt_bboxes
            pos_bbox_targets = self.bbox_coder.encode(
                pos_bboxes, pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            bbox_targets_maxone[:num_pos, :] = pos_bbox_targets_maxone
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, bbox_targets_maxone, iou_targets, iou_labels


    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_iou = [res.pos_iou for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights, bbox_targets_maxone, iou_targets, iou_labels = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_iou,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_targets_maxone = torch.cat(bbox_targets_maxone, 0)
            iou_targets = torch.cat(iou_targets, 0)
            iou_labels = torch.cat(iou_labels, 0)
        return labels, label_weights, bbox_targets, bbox_weights, bbox_targets_maxone, iou_targets, iou_labels


    def _bbox_forward(self, x, rois, pos_rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if pos_rois == None:
            bbox_feats_pos = None
        else:
            bbox_feats_pos = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], pos_rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, bbox_feats_pos=bbox_feats_pos)
        return bbox_results


    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        pos_inds = torch.cat([res.pos_inds for res in sampling_results])
        # print("posinds")
        # print(len(pos_inds))
        rois = bbox2roi([res.bboxes for res in sampling_results])
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # if pos_rois.shape[0] == 0:
        #     return dict(loss_bbox_iou=None)
        bbox_results = self._bbox_forward(x, rois, pos_rois)
        bbox_targets = self.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets[0:4])
        bbox_results.update(loss_bbox=loss_bbox)
        if pos_rois.shape[0] == 0:
            return bbox_results
        labels = bbox_targets[0]
        pos_ind = (labels >= 0) & (labels < self.num_classes)
        pos_cls_pred = bbox_results['cls_score'][pos_ind]
        pos_bbox = bbox_results['bbox_pred'][pos_ind]
        bbox_iou_pred = self.bbox_iou_head(bbox_results['bbox_feats_pos'],
                                           pos_cls_pred, pos_bbox)
        bbox_preds = self.bbox_coder.decode(rois[:, 1:], bbox_results['bbox_pred'])
        pos_bbox_pred = bbox_preds.view(-1, bbox_preds[0].size(-1))[pos_ind]
        pos_bbox_target = bbox_targets[-3].view(-1, bbox_targets[-3][0].size(-1))[pos_ind]

        bbox_iou_targets = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
        # bbox_iou_target = torch.cat([res.pos_iou for res in sampling_results])
        # bbox_iou_targets = bbox_targets[-2]
        # bbox_iou_targets = ((2 * bbox_iou_targets - 1) * 5).ceil()
        # inds = bbox_iou_targets < 0
        # bbox_iou_targets[inds] = 5
        # print(bbox_iou_targets)


        loss_bbox_iou = self.bbox_iou_head.loss(bbox_iou_pred.squeeze(-1),
                                                bbox_iou_targets)
        # print(loss_bbox_iou)
        # print(bbox_results['loss_bbox'])
        bbox_results['loss_bbox'].update(loss_bbox_iou)
        return bbox_results


    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        pos_rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois, pos_rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        iou_score = self.bbox_iou_head(bbox_results['bbox_feats_pos'], bbox_results['cls_score'], bbox_results['bbox_pred'])
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        iou_score = iou_score.split(num_proposals_per_img, 0)


        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some Detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_iou_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                iou_score[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels


    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))


    # def update_hyperparameters(self):
    #     """Update hyperparameters like IoU thresholds for assigner and beta for
    #     SmoothL1 loss based on the training statistics.
    #
    #     Returns:
    #         tuple[float]: the updated ``iou_thr`` and ``beta``.
    #     """
    #     new_rcnn_iou_thr = max(0.4,
    #                       np.mean(self.iou_history))
    #     self.iou_history = []
    #     self.bbox_assigner.pos_iou_thr = new_rcnn_iou_thr
    #     self.bbox_assigner.neg_iou_thr = new_rcnn_iou_thr
    #     self.bbox_assigner.min_pos_iou = new_rcnn_iou_thr
    #     # new_beta = min(self.train_cfg.dynamic_rcnn.initial_beta,
    #     #                np.median(self.beta_history))
    #     # self.beta_history = []
    #     # self.bbox_head.loss_bbox.beta = new_beta
    #     print("new_rcnn_iou_thr")
    #     print(new_rcnn_iou_thr)
    #     return new_rcnn_iou_thr
    #     # return new_iou_thr







