from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
import torch
@BBOX_CODERS.register_module()
class VMANMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
    """

    def __init__(self,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, 
                      cls_scores, 
                      pts_preds,
                      attrs_preds,
                      img_metas):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """

        # use score threshold to get mask
        
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        pts_preds = pts_preds[bbox_index]

        img_shape = img_metas['img_shape']
        final_pts_preds = pts_preds.clone()
        final_pts_preds[..., 0:1] = pts_preds[..., 0:1] * img_shape[0]
        final_pts_preds[..., 1:2] = pts_preds[..., 1:2] * img_shape[1] #num_q,num_p,2

        final_scores = scores 
        final_preds = labels 

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score
        else:
            thresh_mask = torch.ones(final_scores.shape, dtype=bool)
        scores = final_scores[thresh_mask]
        pts = final_pts_preds[thresh_mask]
        labels = final_preds[thresh_mask]

        if attrs_preds is not None:
            all_attrs_scores = []
            all_attrs_labels = []
            for attr_scores in attrs_preds:
                attr_scores = attr_scores.squeeze()
                attr_scores = attr_scores.softmax(1)
                _, attr_labels = torch.max(attr_scores[bbox_index, :], 1)
                all_attrs_labels.append(attr_labels[thresh_mask].cpu())
                all_attrs_scores.append(attr_scores[bbox_index][thresh_mask].cpu())

            final_attrs = {'attrs_scores':all_attrs_scores, 'attrs_preds':all_attrs_labels}
        else:
            final_attrs = None
            
        predictions_dict = {
            'scores': scores,
            'labels': labels,
            'pts': pts,
            'attrs':final_attrs
        }
        return predictions_dict

    def decode(self, preds_dicts, img_metas):
        """Decode bboxes.
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        # all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]
        all_attrs_preds = preds_dicts['all_attrs_preds'][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], 
                                                       all_pts_preds[i],
                                                       all_attrs_preds, 
                                                       img_metas[i]))
        return predictions_list