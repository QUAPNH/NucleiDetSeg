_base_ = '../Mask_Rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    type='BBoxScoringRCNN',
    roi_head=dict(
        type='BBoxScoringRoIHead',
        num_classes=1,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_decoded_bbox=False,
        bbox_iou_head=dict(
            type='BBoxIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=7,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1)))
# model training and testing settings
# train_cfg = dict(rcnn=dict(mask_thr_binary=0.5))
