from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
@DETECTORS.register_module()
class VMA(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(VMA,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

    def extract_img_feat(self, 
                         img, 
                         ):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        return img_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, 
                     img, 
                     img_metas=None, 
                     ):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_labels, 
                          gt_bboxes,
                          gt_attrs,
                          img_metas,
                          ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_labels (list[torch.Tensor]): Ground truth labels for
                boxes of each sample
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes (list[torch.Tensor], optional): Ground truth
                boxes.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_labels, gt_bboxes, gt_attrs, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      img=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_attrs=None,
                      img_metas=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats = self.extract_feat(img=img)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_labels, gt_bboxes, gt_attrs, img_metas)

        losses.update(losses_pts)
        return losses

    def forward_test(self, 
                    img_metas, 
                    img=None, 
                    rescale=None,
                    **kwargs,
                    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        bbox_results = self.simple_test(
            img_metas, img, **kwargs)
        for idx in range(len(bbox_results)):
            bbox_results[idx]['pts_bbox']['img_metas'] = img_metas[idx]
            for key, value in kwargs.items():
                bbox_results[idx]['pts_bbox'][key] = value[idx]
        return bbox_results

    def pred2result(self, 
                    scores, 
                    labels, 
                    pts, 
                    attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            pts (torch.Tensor): Points with shape of (n, 2).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs

        return result_dict
    
    def simple_test_pts(self, 
                        x, 
                        img_metas, 
                        ):
        """Test function"""
        
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, 
                                                  img_metas, 
                                                  )
        bbox_results = [
            self.pred2result(scores, labels, pts, attrs)
            for scores, labels, pts, attrs in bbox_list
        ]
        return bbox_results
    def simple_test(self, 
                    img_metas, 
                    img=None, 
                    **kwargs
                    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list