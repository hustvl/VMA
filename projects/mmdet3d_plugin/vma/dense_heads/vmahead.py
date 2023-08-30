
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core import (multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version
from ..builder import build_attr_head
@HEADS.register_module()
class VMAHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 num_vec=20,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 loss_pts2lines=dict(type='ChamferDistance', 
                             loss_src_weight=1.0, 
                             loss_dst_weight=1.0),
                 loss_pts2pts=dict(type='ChamferDistance', 
                             loss_src_weight=1.0, 
                             loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_attr=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 **kwargs):

        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_cls_fcs = num_cls_fcs - 1
        
        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        if 'attr_head' in kwargs:
            self.attr_head_cfg = kwargs['attr_head']
        else:
            self.attr_head_cfg = None
        super(VMAHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.loss_pts2lines = build_loss(loss_pts2lines)
        self.loss_dir = build_loss(loss_dir)
        self.loss_pts2pts = build_loss(loss_pts2pts)
        self.loss_attr = build_loss(loss_attr)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        if self.attr_head_cfg is not None:
            attr_head = build_attr_head(self.attr_head_cfg)

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            if self.attr_head_cfg is not None:
                self.attr_head_branches = _get_clones(attr_head, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            if self.attr_head_cfg is not None:
                self.attr_head_branches = nn.ModuleList(
                    [attr_head for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], 0.)
    
    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, 
                mlvl_feats, 
                img_metas=None, 
                ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, _, H, W = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if hasattr(self.transformer, 'encoder'):
            input_img_h, input_img_w = img_metas[0]['img_shape']
            img_masks = mlvl_feats[0].new_ones((bs, input_img_h, input_img_w))
            for img_id in range(bs):
                img_h, img_w = img_metas[img_id]['img_shape']
                img_masks[img_id, :img_h, :img_w] = 0

            mlvl_masks = []
            mlvl_positional_encodings = []
            for feat in mlvl_feats:
                mlvl_masks.append(F.interpolate(img_masks[None],size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1])) 
        else:
            mlvl_masks=None
            mlvl_positional_encodings=None

        outputs = self.transformer(
            mlvl_feats,
            mlvl_masks,
            object_query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )
    
        hs, init_reference, inter_references, _, _ = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_pts_coords = [] 
        outputs_attrs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl]
                                            .view(bs,self.num_vec, self.num_pts_per_vec,-1)
                                            .mean(2)) 
            if self.attr_head_cfg is not None:
                outputs_attr = self.attr_head_branches[lvl](hs[lvl]
                                                .view(bs, self.num_vec, self.num_pts_per_vec,-1)
                                                .mean(2))
            else:
                outputs_attr = None
            tmp = self.reg_branches[lvl](hs[lvl]) 
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid() 

            outputs_pts_coord = tmp.view(tmp.shape[0], self.num_vec,
                                self.num_pts_per_vec,2)
            outputs_pts_coord = torch.clamp(outputs_pts_coord, min=0, max=0.999)
            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(outputs_pts_coord)
            outputs_attrs.append(outputs_attr)
        outputs_classes = torch.stack(outputs_classes)
        outputs_pts_coords = torch.stack(outputs_pts_coords)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_pts_preds': outputs_pts_coords,
            'all_attrs_preds':outputs_attrs,
            'enc_cls_scores': None,
            'enc_pts_preds': None
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           pts_pred,
                           attrs_preds, 
                           gt_labels,
                           gt_shifts_pts,
                           gt_attrs,
                           ):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            pts_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = pts_pred.size(0)
        assign_result, order_index = self.assigner.assign(cls_score, pts_pred, gt_labels, gt_shifts_pts,)
        pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
                                              gt_shifts_pts,)

        # pos_inds = sampling_result.pos_inds
        # neg_inds = sampling_result.neg_inds
        pos_inds = pts_sampling_result.pos_inds
        neg_inds = pts_sampling_result.neg_inds
        pos_assigned_gt_inds = pts_sampling_result.pos_assigned_gt_inds

        # attr targets
        if gt_attrs is not None:
            num_attrs = len(gt_attrs)
            assigned_attrs_labels = [gt_attrs[i][pos_assigned_gt_inds] for i in range(num_attrs)]
            pos_attrs_pred = [attrs_preds[i][pos_inds] for i in range(num_attrs)]
        else:
            assigned_attrs_labels = None
            pos_attrs_pred = None

        # label targets
        labels = gt_shifts_pts.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pts_sampling_result.pos_assigned_gt_inds]
        label_weights = gt_shifts_pts.new_ones(num_bboxes)

        # pts targets
        pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            assigned_shift = gt_labels[pts_sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[pts_sampling_result.pos_inds, pts_sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        pts_targets[pos_inds] = gt_shifts_pts[pts_sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, pts_targets, pts_weights, assigned_attrs_labels, pos_attrs_pred, pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    pts_preds_list,
                    attrs_preds_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_attrs_list, 
                    ):
        
        (labels_list, label_weights_list, pts_targets_list, pts_weights_list, assigned_attrs_labels_list, pos_attrs_pred_list, 
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, pts_preds_list, attrs_preds_list, gt_labels_list, gt_shifts_pts_list, gt_attrs_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, pts_targets_list, pts_weights_list,assigned_attrs_labels_list, pos_attrs_pred_list, 
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    pts_preds,
                    attrs_preds,
                    gt_labels_list,
                    gt_attr_list,
                    gt_shifts_pts_list,
                    ):
        
        # num_imgs = cls_scores.size(0)
        num_imgs = pts_preds.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        
        if attrs_preds is not None:
            attr_preds_list = []
            num_attrs = len(attrs_preds)
            for i in range(num_imgs):
                attr_preds_list.append([attrs_preds[j][i] for j in range(num_attrs)])
        else:
            attr_preds_list = [None for _ in range(num_imgs)]
            gt_attr_list = [None for _ in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, pts_preds_list, attr_preds_list, gt_labels_list, gt_shifts_pts_list, gt_attr_list)
        (labels_list, label_weights_list, pts_targets_list, pts_weights_list, assigned_attrs_labels_list, pos_attrs_pred_list, num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)
        
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        
        if attrs_preds is not None:
            assigned_attrs_labels = []
            pos_attrs_preds = []
            for i in range(num_attrs):
                assigned_attrs_labels.append(torch.cat([assigned_attrs_labels_list[j][i] for j in range(num_imgs)], 0))
                pos_attrs_preds.append(torch.cat([pos_attrs_pred_list[j][i] for j in range(num_imgs)], 0))
            loss_attr=torch.tensor([0.0], device = torch.device('cuda'))
            for assigned_attr_labels, pos_attr_preds in zip(assigned_attrs_labels, pos_attrs_preds):
                attr_label_weights = assigned_attr_labels.new_ones(assigned_attr_labels.shape, dtype=torch.long)
                loss_attr += self.loss_attr(
                    pos_attr_preds, assigned_attr_labels, attr_label_weights, avg_factor=cls_avg_factor)
        else:
            loss_attr = None
        
        
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
      
        normalized_pts_targets = pts_targets 
        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        loss_pts2lines = self.loss_pts2lines(
            pts_preds, 
            normalized_pts_targets, 
            pts_weights,
            avg_factor=num_total_pos)
        loss_pts2pts = self.loss_pts2pts(
            pts_preds, 
            normalized_pts_targets, 
            pts_weights,
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        denormed_pts_preds = pts_preds
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir, 
            pts_targets_dir,
            dir_weights,
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_pts2lines = torch.nan_to_num(loss_pts2lines)
            loss_pts2pts = torch.nan_to_num(loss_pts2pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_pts2lines, loss_pts2pts, loss_dir, loss_attr

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_labels_list,
             gt_bboxes_list,
             gt_attrs_list,
             preds_dicts,
             img_metas=None
             ):
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        all_cls_scores = preds_dicts['all_cls_scores']
        all_pts_preds  = preds_dicts['all_pts_preds']
        all_attrs_preds = preds_dicts['all_attrs_preds']

        device = gt_labels_list[0].device
        gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        
        num_dec_layers = len(all_pts_preds)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_list = [gt_attrs_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        
        losses_cls, losses_pts2lines, losses_pts2pts, losses_dir, losses_attr= multi_apply(
            self.loss_single, all_cls_scores, all_pts_preds, all_attrs_preds, all_gt_labels_list, all_gt_attr_list, all_gt_shifts_pts_list) # 这里输入的是所有的pred_pts和gt_pts，但是是对每一个decoder layer来计算loss
        
        loss_dict = dict()

        # loss from the last decoder layer
        
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_attr'] = losses_attr[-1]
        loss_dict['loss_pts2lines'] = losses_pts2lines[-1]
        loss_dict['loss_pts2pts'] = losses_pts2pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_attr_i, loss_pts2lines_i, loss_pts2pts_i, loss_dir_i in zip(losses_cls[:-1],
                                                                            losses_attr[:-1],
                                                                            losses_pts2lines[:-1],
                                                                            losses_pts2pts[:-1],
                                                                            losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_attr'] = loss_attr_i
            loss_dict[f'd{num_dec_layer}.loss_pts2lines'] = loss_pts2lines_i
            loss_dict[f'd{num_dec_layer}.loss_pts2pts'] = loss_pts2pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        
        if gt_attrs_list is None:
            key_list = list(loss_dict.keys())
            for key in key_list:
                if 'loss_attr' in key:
                    loss_dict.pop(key)
        
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts, img_metas)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']
            attrs = preds['attrs']
            ret_list.append([scores, labels, pts, attrs])

        return ret_list