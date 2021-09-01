import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, MaxPool2d, kaiming_init, normal_init, ConvModule, build_upsample_layer
from mmcv.runner import force_fp32
from torch.nn.modules.utils import _pair
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms, multiclass_nms_score

from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class BBoxIoUHead(nn.Module):
    """Mask IoU Head.

    This head predicts the IoU of predicted masks and corresponding gt masks.
    """

    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=80,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 loss_iou=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)):
                 # loss_iou=dict(type='MSELoss', loss_weight=0.5)):
        super(BBoxIoUHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.fp16_enabled = False
        bbox_coder = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2])
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                # concatenation of mask feature and mask prediction
                in_channels = self.in_channels
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(
                Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=stride,
                    padding=1))
        upsample_in_channels = in_channels
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        roi_feat_size = _pair(roi_feat_size)
        pooled_area = (roi_feat_size[0] ) * (roi_feat_size[1] )
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            if i == 0:
                in_channels = self.conv_out_channels * pooled_area
            elif i==1:
                in_channels = self.fc_out_channels
                # in_channels = self.fc_out_channels + 6
            else:
                in_channels = self.fc_out_channels
            self.fcs.append(Linear(in_channels, self.fc_out_channels))

        self.fc_bbox_iou = Linear(self.fc_out_channels, 1)
        self.relu = nn.ReLU()
        self.max_pool = MaxPool2d(2, 2)
        self.loss_iou = build_loss(loss_iou)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(
                fc,
                a=1,
                mode='fan_in',
                nonlinearity='leaky_relu',
                distribution='uniform')
        normal_init(self.fc_bbox_iou, std=0.01)

    def forward(self, bbox_feat, cls_pred, bbox_pred):
        cls_pred = cls_pred.sigmoid()
        bbox_pred = bbox_pred
        # mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))
        # x = torch.cat((bbox_feat, mask_pred_pooled), 1)
        if self.upsample is not None:
            bbox_feat = self.upsample(bbox_feat)
            if self.upsample_method == 'deconv':
                bbox_feat = self.relu(bbox_feat)
        # print("x1")
        # print(bbox_feat.shape)
        for conv in self.convs:
            bbox_feat = self.relu(conv(bbox_feat))
        x = bbox_feat.flatten(1)
        for i, fc in enumerate(self.fcs):
            x = self.relu(fc(x))
            # if i == 0:
            #     x = torch.cat((x, cls_pred, bbox_pred), 1)
        bbox_iou = self.fc_bbox_iou(x)
        return bbox_iou

    @force_fp32(apply_to=('mask_iou_pred', ))
    def loss(self, bbox_iou_pred, bbox_iou_targets):
        pos_inds = bbox_iou_targets > 0
        # print(bbox_iou_pred[pos_inds].shape)
        # print(bbox_iou_targets[pos_inds].shape)
        if pos_inds.sum() > 0:
            loss_bbox_iou = self.loss_iou(bbox_iou_pred[pos_inds],
                                          bbox_iou_targets[pos_inds])
        else:
            loss_bbox_iou = bbox_iou_pred.sum() * 0
        return dict(loss_bbox_iou=loss_bbox_iou)


    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'iou_score'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   iou_score,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if isinstance(iou_score, list):
            iou_score = sum(iou_score) / float(len(iou_score))
        # nms_pre = cfg.get('nms_pre', -1)
        cls_scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        # iou_score = F.softmax(iou_score, dim=1) if iou_score is not None else None
        # iou_scores = torch.zeros([iou_score.size(0), 1]).cuda()
        # print(iou_score[:, :1])
        # for i in range(5):
        #     a = 1/2 + i/10
        #     print(iou_score[:, :(i+1)])
            # iou_scores += a * iou_score[:, i:(i+1)]
        # max_scores, _ = (cls_scores[:, :-1] * iou_score).sqrt().max(dim=1)
        # _, topk_inds = max_scores.topk(nms_pre)
        # bbox_pred = bbox_pred[topk_inds, :]
        # cls_scores = cls_scores[topk_inds]
        # iou_scores = iou_score[topk_inds]
        # iou_scores = iou_score
        iou_scores = iou_score.sigmoid()


        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, cls_scores, iou_scores
        else:
            det_bboxes, det_labels = multiclass_nms_score(bboxes, cls_scores, iou_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            # det_bboxes, det_labels = self.score_voting(det_bboxes, det_labels,
            #                                            bboxes,
            #                                            nms_scores,
            #                                            cfg.score_thr)

            return det_bboxes, det_labels


