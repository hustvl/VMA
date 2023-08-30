import cv2
import numpy as np
import os
import json
from collections import defaultdict
from shapely.geometry import LineString, Polygon, GeometryCollection
from shapely.strtree import STRtree
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance
from utils import DouglasPeuker_v3

def combine_results_and_save_driving_freespace(config, img, results, out_dir=None, left_top=None, visualize_flag=False):
    print('Start to aggregate driving freespace instances')
    image_name = img.split('/')[-1]

    freespace_class = config['map_classes']
    # first step: nms to deduplicated
    aggr_pred_instances = []
    big_results = []
    for i in range(len(results)):
        pred_instances = results[i]
        left, top = left_top[i]
        for pred_instance in pred_instances:
            instance_data = [[x[0]*1000+left, x[1]*1000+top] for x in pred_instance['data']] if (pred_instance['data']<1).all() else [[x[0]+left, x[1]+top] for x in pred_instance['data']]
            big_results.append({'type':freespace_class[pred_instance['class']], 'seq':instance_data, 'confidence_level':pred_instance['confidence_level']})
    cls_gens = format_res_gt_by_classes_freespace(big_results,
                                            cls_names=freespace_class,
                                            num_pred_pts_per_instance=5)
    for i, clsname in enumerate(freespace_class):
    
        # get gt and det bboxes of this class
        cls_gen = cls_gens[clsname]
        if cls_gen.shape[0] == 0:
            continue
        
        cls_aggr_gen = gen_distinct_instances(cls_gen, cls_gen, threshold=20, metric='chamfer', num_attrs=2)
        cls_aggr_gen = gen_union_instances(cls_aggr_gen[:,:-2], cls_aggr_gen[:,:-2], iou_threshold=0.1, metric='iou')
        cls_aggr_gen = gen_union_instances(cls_aggr_gen, cls_aggr_gen, iou_threshold=0.01, metric='iou')
        for single_cls_aggr_gen in cls_aggr_gen:
            data = np.array(single_cls_aggr_gen).reshape(-1,2).tolist()
            d = DouglasPeuker_v3(2, 10)
            data = d.main(data)
            aggr_pred_instances.append({'type':clsname, 'seq':data})
    print('driving freespace aggregated!')
    vector_outdir = os.path.join(out_dir, 'vector_out')
    if not os.path.exists(vector_outdir):
        os.makedirs(vector_outdir)
        
    json_path = os.path.join(vector_outdir, image_name.replace('.jpg', '_driving_freespace.json'))
    with open(json_path, 'w') as fout:
        json.dump(aggr_pred_instances, fout)
    if visualize_flag:
        visualize_aggregated_out_dir = os.path.join(out_dir, 'visualize')
        if not os.path.exists(visualize_aggregated_out_dir):
            os.makedirs(visualize_aggregated_out_dir)
        visualize_single_image_freespace(img, aggr_pred_instances, visualize_aggregated_out_dir)

def visualize_single_image_freespace(image_path, pred_instances, out_dir):
    # import pdb;pdb.set_trace()
    image = cv2.imread(image_path)
    drew_image_path = '/'.join([out_dir, image_path.split('/')[-1].replace('.jpg', '_driving_freespace.jpg')])
    for pred_instance in pred_instances:
        sub_pred_points = [[round(x[0]), round(x[1])]for x in pred_instance['seq']]
        sub_pred_points += [sub_pred_points[0]]
        color = (0,0,255)
        for i in range(len(sub_pred_points)): 
            if i == 0:
                cv2.circle(image, sub_pred_points[i], 15, (0,97,255), -1)#中心坐标,半径,颜色(BGR),线宽(若为-1,即为填充颜色)s
                continue
            elif i == len(sub_pred_points)-1:
                cv2.line(image, sub_pred_points[i-1], sub_pred_points[i], color, 4)
                cv2.circle(image, sub_pred_points[i], 10, (0,255,255), -1)#中心坐标,半径,颜色(BGR),线宽(若为-1,即为填充颜色)s
            else:
                cv2.line(image, sub_pred_points[i-1], sub_pred_points[i], color, 4)
                cv2.circle(image, sub_pred_points[i], 8, (0,255,0), -1)
    cv2.imwrite(drew_image_path, image)

def gen_union_instances(gen_boxes,
                        gt_boxes,
                        line_width=1.,
                        iou_threshold=0.1,
                        metric='iou',
                        ):
    if isinstance(gen_boxes, list):
        num_gens = len(gen_boxes)
        num_gts = len(gt_boxes)
    else:
        num_gens = gen_boxes.shape[0]
        num_gts = gt_boxes.shape[0]
        gen_boxes = gen_boxes.reshape(num_gens,-1,2).tolist()
        gt_boxes = gt_boxes.reshape(num_gts,-1,2).tolist()
    
    # iou matrix: n x m
    matrix = custom_polyline_score_distinct(gen_boxes, 
                                            gt_boxes,
                                            linewidth=line_width,
                                            metric=metric)
    aggregation_index_0, aggregation_index_1 = np.where(matrix>iou_threshold)
    if not len(aggregation_index_0):
        return gen_boxes
    unionset_list = get_union(aggregation_index_0.tolist(), aggregation_index_1.tolist())
    delete_instances = []
    for union_set in unionset_list:
        union = list(union_set)
        for index in range(1, len(union)):
            delete_instances.append(gen_boxes[union[index]])
            union_polygon = Polygon(gen_boxes[union[0]]).buffer(0.01).union(Polygon(gen_boxes[union[index]]).buffer(0.01))
            if isinstance(union_polygon, Polygon):
                gen_boxes[union[0]] = list(union_polygon.exterior.coords)
            elif isinstance(union_polygon, GeometryCollection):
                for instance_polygon in union_polygon:
                    if isinstance(instance_polygon, Polygon):
                        gen_boxes[union[0]] = list(instance_polygon.exterior.coords)
                        continue
            # else:
            #     print(union_polygon)
    for delete_instance in delete_instances:
        if delete_instance in gen_boxes:
            gen_boxes.remove(delete_instance)
    return gen_boxes

def get_union(x_list, y_list):
    pair_list = [[x, y] for x, y in zip(x_list, y_list)]
    final_list = []
    # import pdb;pdb.set_trace()
    for i in range(len(pair_list)):
        a = set(pair_list[i])
        for j in range(1, len(pair_list)):
            b = set(pair_list[j])
            if a&b:
                a = a|b
        if a not in final_list:
            final_list.append(a)
    # print(final_list)
    # import pdb;pdb.set_trace()
    return final_list


def gen_distinct_instances(gen_lines,
                           gt_lines,
                           line_width=1.,
                           threshold=2,
                           metric='chamfer',
                           num_attrs=0):
    # import pdb;pdb.set_trace()
    if metric == 'chamfer':
        if threshold >0:
            threshold= -threshold

    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]
    gen_scores = gen_lines[:,-1] # n
    
    # distance matrix: n x m
    matrix = custom_polyline_score_distinct(
            gen_lines.reshape(num_gens,-1,2), 
            gt_lines.reshape(num_gts,-1,2),
            linewidth=line_width,
            metric=metric)

    aggregation_index_0, aggregation_index_1 = np.where(matrix>threshold)
    delete_indexes = []
    if not len(aggregation_index_0):
        return gen_lines
    for i in range(len(aggregation_index_0)):
        if aggregation_index_0[i] in delete_indexes or aggregation_index_1[i] in delete_indexes:
            continue
        if gen_scores[aggregation_index_0[i]] >= gen_scores[aggregation_index_1[i]]:
            delete_index = aggregation_index_1[i]
        else:
            delete_index = aggregation_index_0[i]
        if delete_index not in delete_indexes:
            delete_indexes.append(delete_index)
    if len(delete_indexes):
        gen_lines_distinct_v0 = np.delete(gen_lines, delete_indexes,0)
    else:
        gen_lines_distinct_v0 = gen_lines

    return gen_lines_distinct_v0

def custom_polyline_score_distinct(pred_lines, 
                                   gt_lines, 
                                   linewidth=1., 
                                   metric='chamfer'):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''
    if metric == 'iou':
        linewidth = 1.0
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    pred_lines_shapely = \
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                          for i in pred_lines]
    gt_lines_shapely =\
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                        for i in gt_lines]
    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    if metric=='chamfer':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    elif metric=='iou':
        iou_matrix = np.zeros((num_preds, num_gts),dtype=np.float64)
    else:
        raise NotImplementedError

    for i, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]
                if metric=='chamfer':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], 'euclidean')
                    valid_ab = dist_mat.min(-1).mean()
                    valid_ba = dist_mat.min(-2).mean()
                    iou_matrix[pred_id, i] = -(valid_ba+valid_ab)/2
                    if pred_id == i:
                        iou_matrix[pred_id, i] = -100.0
                elif metric=='iou':
                    inter = o.intersection(pline).area
                    union = o.union(pline).area
                    iou_matrix[pred_id, i] = inter / union
                    if pred_id == i:
                        iou_matrix[pred_id, i] = 0.0
    return iou_matrix

def format_res_gt_by_classes_freespace(gen_results,
                                 cls_names=None,
                                 num_pred_pts_per_instance=5,
                                 num_attrs=0):
    assert cls_names is not None

    cls_gens= defaultdict(dict)
    for i, clsname in enumerate(cls_names):
        cls_gen, cls_score, cls_attrs = [], [], []
        for gen_result in gen_results:
            if gen_result['type'] == clsname:
                gen_result['seq'] = [[x[0]*1000, x[1]*1000] for x in gen_result['seq']] if (np.array(gen_result['seq'])<1).all() else [[x[0], x[1]] for x in gen_result['seq']]
                sampled_points = np.array(gen_result['seq'])
                cls_gen.append(sampled_points)
                cls_score.append(gen_result['confidence_level'])
        num_res = len(cls_gen)
        if num_res > 0:
            cls_gen = np.stack(cls_gen).reshape(num_res,-1)
            cls_score = np.array(cls_score)[:,np.newaxis]
        else:
            cls_gen = np.zeros((0,num_pred_pts_per_instance*2+1+num_attrs))
        cls_gens[clsname] = cls_gen
    return cls_gens

