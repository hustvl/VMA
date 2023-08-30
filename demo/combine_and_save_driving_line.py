import torch
import cv2
import numpy as np
import os
import json
from collections import defaultdict
from shapely.geometry import LineString, CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from scipy.spatial import distance
from utils import DouglasPeuker_v3

def combine_results_and_save_driving_line(config, img, results, out_dir=None, left_top=None, visualize_flag=False):
    print('Start to aggregate driving line instances')

    image_name = img.split('/')[-1]
    line_class = config['map_classes']
    lane_direction = config['lane_direction']
    lane_type = config['lane_type']
    lane_properties = config['lane_properties']
    lane_flag = config['lane_flag']
    lane_width = config['lane_width']
    curb_type = config['curb_type']
    lane_start_idx = 0
    curb_start_idx = len(lane_direction) + len(lane_type) + \
        len(lane_properties) + len(lane_flag) + len(lane_width)
    distinct_pred_instances_big_image = []
    metric = 'chamfer'

    # first step: dedulipcate in one 1k image
    for pred_instances in results:
        cls_gens = format_res_by_classes_line_resample(pred_instances,
                                                        cls_names=line_class,
                                                        num_sample=600,
                                                        num_pred_pts_per_instance=600,
                                                        eval_use_same_gt_sample_num_flag=True, 
                                                        fix_interval=False)
        distinct_threshold = 2
        distinct_pred_instances_small_image = []
        for i, clsname in enumerate(line_class):
            cls_gen = cls_gens[clsname]
            if cls_gen['data'].shape[0] == 0:
                continue
            linewidth = 1.
            # deduplicate
            cls_gen_distinct = gen_distinct_instances(cls_gen,
                                                      cls_gen,
                                                      line_width=linewidth, 
                                                      distance_threshold=distinct_threshold, 
                                                      metric=metric)
            for j in range(cls_gen_distinct['data'].shape[0]):
                data = cls_gen_distinct['data'][j].reshape(-1, 2)
                confidence_level = cls_gen_distinct['confidence_level'][j]
                attrs_labels = cls_gen_distinct['attrs_labels'][j]
                attrs_scores = cls_gen_distinct['attrs_scores'][j]
                distinct_pred_instances_small_image.append({'class':i, 'data':data, 'confidence_level':confidence_level, 'attrs_labels':attrs_labels, 'attrs_scores':attrs_scores})
        distinct_pred_instances_big_image.append(distinct_pred_instances_small_image)
    
    aggregated_big_image_instances = []
    aggregated_instances = distinct_pred_instances_big_image[0]
    left_top_0=left_top[0]
    classes_aggregated = format_res_by_classes_line(aggregated_instances,
                                                    cls_names=line_class,
                                                    left_top=left_top_0)
    length_threshold=30
    distance_threshold=4.
    for i in range(1, len(distinct_pred_instances_big_image)):
        # print(i)
        unaggregated_instances = distinct_pred_instances_big_image[i]
        classes_unaggregated = format_res_by_classes_line(unaggregated_instances,
                                                          cls_names=line_class,
                                                          left_top=left_top[i])
        for clsname in line_class:
            class_aggregated = classes_aggregated[clsname]
            class_unaggregated = classes_unaggregated[clsname]
            aggregated_lines = class_aggregated['data']
            unaggregated_lines = class_unaggregated['data']
            aggregated_lines_scores = class_aggregated['confidence_level']
            unaggregated_lines_scores = class_unaggregated['confidence_level']
            aggregated_lines_attrs_scores = class_aggregated['attrs_scores']
            unaggregated_lines_attrs_scores = class_unaggregated['attrs_scores']
            if len(unaggregated_lines) == 0:
                continue
            elif len(aggregated_lines) == 0 and len(unaggregated_lines) != 0:
                classes_aggregated[clsname]['data'] = unaggregated_lines
                classes_aggregated[clsname]['confidence_level'] = unaggregated_lines_scores
                classes_aggregated[clsname]['attrs_scores'] = unaggregated_lines_attrs_scores
                continue
            linewidth = 1.
            classes_aggregated[clsname] = gen_aggregated_instances_with_attr(aggregated_lines, 
                                                                             unaggregated_lines,
                                                                             aggregated_lines_scores,
                                                                             unaggregated_lines_scores, 
                                                                             aggregated_lines_attrs_scores, 
                                                                             unaggregated_lines_attrs_scores,
                                                                             line_width=2.0, 
                                                                             distance_threshold=distance_threshold, 
                                                                             length_threshold=length_threshold, 
                                                                             metric=metric,
                                                                             image_path=img,
                                                                             image_idx=i,
                                                                             clsname=clsname,)
    for i, clsname in enumerate(line_class):
        if len(classes_aggregated[clsname]['data'])==0:
            continue
        
        class_aggregated = classes_aggregated[clsname]
        aggregated_lines = [class_aggregated['data'][0]]
        unaggregated_lines = class_aggregated['data'][1:]
        aggregated_lines_scores = class_aggregated['confidence_level'][0][np.newaxis,:]
        unaggregated_lines_scores = class_aggregated['confidence_level'][1:]
        aggregated_lines_attrs_scores = class_aggregated['attrs_scores'][0][np.newaxis,:]
        unaggregated_lines_attrs_scores = class_aggregated['attrs_scores'][1:]
        classes_aggregated[clsname] = gen_aggregated_instances_with_attr(aggregated_lines, 
                                                                         unaggregated_lines,
                                                                         aggregated_lines_scores,
                                                                         unaggregated_lines_scores, 
                                                                         aggregated_lines_attrs_scores, 
                                                                         unaggregated_lines_attrs_scores,
                                                                         line_width=2.0, 
                                                                         distance_threshold=distance_threshold, 
                                                                         length_threshold=length_threshold, 
                                                                         metric=metric,
                                                                         image_path=img,
                                                                         clsname=clsname,)
    
    for cls_name in line_class:
        class_aggregated = classes_aggregated[cls_name]
        class_aggregated_data = class_aggregated['data']
        class_aggregated_attrs = class_aggregated['attrs_scores']
        if cls_name == 'lane':
            for single_cls_gen_aggregated_data, single_cls_gen_aggregated_attrs in zip(class_aggregated_data, class_aggregated_attrs):
                # import pdb;pdb.set_trace()
                attrs = []
                cur_idx = 0
                for lane_attr_class in [lane_direction, lane_type, lane_properties, lane_flag, lane_width]:
                    attrs.append(lane_attr_class[np.argmax(
                        single_cls_gen_aggregated_attrs[
                            lane_start_idx+cur_idx: lane_start_idx+cur_idx+len(lane_attr_class)
                            ]
                        )])
                    cur_idx += len(lane_attr_class)
                data = single_cls_gen_aggregated_data if isinstance(single_cls_gen_aggregated_data, list) else single_cls_gen_aggregated_data.tolist()
                d = DouglasPeuker_v3(2, 10)
                data = d.main(data)[::-1]
                aggregated_big_image_instances.append({'type':cls_name, 'seq':data, 'attrs':attrs})
        elif cls_name == 'curb':
            for single_cls_gen_aggregated_data, single_cls_gen_aggregated_attrs in zip(class_aggregated_data, class_aggregated_attrs):
                # print(curb_start_idx)
                attrs = [curb_type[np.argmax(single_cls_gen_aggregated_attrs[curb_start_idx:])]]
                # attrs = [curb_type[np.argmax(single_cls_gen_aggregated_attrs[21:])]]
                data = single_cls_gen_aggregated_data if isinstance(single_cls_gen_aggregated_data, list) else single_cls_gen_aggregated_data.tolist()
                d = DouglasPeuker_v3(2, 10)
                data = d.main(data)[::-1]
                aggregated_big_image_instances.append({'type':cls_name, 'seq':data, 'attrs':attrs})

    print('drving line aggregated!')
    vector_outdir = os.path.join(out_dir, 'vector_out')
    if not os.path.exists(vector_outdir):
        os.makedirs(vector_outdir)
    json_path = os.path.join(vector_outdir, image_name.replace('.jpg', '_driving_line.json'))
    with open(json_path, 'w') as fout:
        json.dump(aggregated_big_image_instances, fout)
    if visualize_flag:
        visulize_out_dir = os.path.join(out_dir, 'visualize')
        if not os.path.exists(visulize_out_dir):
            os.makedirs(visulize_out_dir)
        visualize_single_image_line(img, aggregated_big_image_instances, visulize_out_dir)  

def visualize_single_image_line(image_path, pred_instances, out_dir):
    image = cv2.imread(image_path)
    color = {'lane':(0,0,0), 'curb':(0,0,255), 'stopline':(240, 32, 160)}
    drew_image_path = '/'.join([out_dir, image_path.split('/')[-1]]).replace('.jpg', '_driving_line.jpg')
    for pred_instance in pred_instances:
        pred_points = [[round(x[0]), round(x[1])] for x in pred_instance['seq']]
        pred_class = pred_instance['type']
        color_fill = color[pred_class]
        for i in range(len(pred_points)): 
            if i == 0:
                cv2.circle(image, pred_points[i], 15, (0,97,255), -1)
            elif i == len(pred_points)-1:
                cv2.line(image, pred_points[i-1], pred_points[i], color_fill, 4)
                cv2.circle(image, pred_points[i], 8, (0,255,255), -1)
            else:
                cv2.line(image, pred_points[i-1], pred_points[i], color_fill, 4)
                cv2.circle(image, pred_points[i], 8, (0,255,0), -1)
    cv2.imwrite(drew_image_path, image)

def format_result(results, score_threshold):
    """Convert the results to the standard format.
    Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str): The prefix of the output jsonfile.
            You can specify the output directory/filename by
            modifying the jsonfile_prefix. Default: None.
    Returns:
        str: Path of the output json file.
    """
    print('Start to convert detection format...')
    results_list = []
    for batch_result in results:
        for result in batch_result:
            result = result['pts_bbox']
            pred_instances = []
            inds = result['scores_3d'] > score_threshold
            pred_scores = result['scores_3d'][inds].numpy()
            pred_data = result['pts_3d'][inds].numpy()
            pred_labels = result['labels_3d'][inds].numpy()
            # import pdb;pdb.set_trace()
            pred_attrs_labels = torch.stack(result['attrs_3d']['attrs_preds'], 1)[inds]
            pred_attrs_scores = torch.cat(result['attrs_3d']['attrs_scores'], 1)[inds]
            for idx in range(len(pred_scores)):
                pred_instances.append({'class':pred_labels[idx], 
                                    'data':pred_data[idx], 
                                    'attrs_labels':pred_attrs_labels[idx],
                                    'attrs_scores':pred_attrs_scores[idx],
                                    'confidence_level':pred_scores[idx]})
            results_list.append(pred_instances)
    print('convert completed')
    return results_list

def format_res_by_classes_line_resample(gen_results,
                                                  cls_names=None,
                                                  num_sample=300,
                                                  num_pred_pts_per_instance=300,
                                                  eval_use_same_gt_sample_num_flag=False,
                                                  fix_interval=False, ):
    assert cls_names is not None
    cls_gens= defaultdict(dict)
    for i, clsname in enumerate(cls_names):
        cls_gen, cls_score, cls_attrs_labels, attrs_scores = [], [], [], []
        for gen_result in gen_results:
            if gen_result['class'] == i:
                gen_result['data'] = [[x[0]*1000, x[1]*1000] for x in gen_result['data']] if (np.array(gen_result['data'])<1).all() else [[x[0], x[1]] for x in gen_result['data']]
                if len(gen_result['data']) < 2:
                    continue
                if not eval_use_same_gt_sample_num_flag:
                    sampled_points = np.array(gen_result['data'])
                else:
                    line = gen_result['data']
                    line = LineString(line)
                    if fix_interval:
                        distances = list(np.arange(1., line.length, 1.))
                        distances = [0,] + distances + [line.length,]
                        sampled_points = np.array([list(line.interpolate(distance).coords)
                                                for distance in distances]).reshape(-1, 2)
                    else:
                        distances = np.linspace(0, line.length, num_sample)
                        sampled_points = np.array([list(line.interpolate(distance).coords)
                                                    for distance in distances]).reshape(-1, 2)
                cls_gen.append(sampled_points)
                cls_score.append(gen_result['confidence_level'])
                cls_attrs_labels.append(gen_result['attrs_labels'])
                attrs_scores.append(gen_result['attrs_scores'])
        num_res = len(cls_gen)
        
        if num_res > 0:
            # import pdb;pdb.set_trace()
            cls_gen = np.stack(cls_gen).reshape(num_res,-1)
            cls_score = np.array(cls_score)[:,np.newaxis]
            attrs_labels = np.stack(cls_attrs_labels)
            attrs_scores = np.stack(attrs_scores)
        else:
            if not eval_use_same_gt_sample_num_flag:
                cls_gen = np.zeros((0,num_pred_pts_per_instance*2))
                cls_score = []
                attrs_labels = []
                attrs_scores = []
            else:
                cls_gen = np.zeros((0,num_pred_pts_per_instance*2))
                cls_score = []
                attrs_labels = []
                attrs_scores = []
        cls_gens[clsname]['data'] = cls_gen
        cls_gens[clsname]['confidence_level'] = cls_score
        cls_gens[clsname]['attrs_labels'] = attrs_labels
        cls_gens[clsname]['attrs_scores'] = attrs_scores
    return cls_gens

def format_res_by_classes_line(gen_results,
                                         cls_names=None,
                                         left_top=(0, 0),
                                         ):
    assert cls_names is not None
    left, top = left_top
    cls_gens= defaultdict(dict)
    for i, clsname in enumerate(cls_names):
        cls_gen, cls_score, attrs_scores = [], [], []
        for gen_result in gen_results:  
            if gen_result['class'] == i:
                gen_result['data'] = [[x[0]*1000+left, x[1]*1000+top] for x in gen_result['data']] if (gen_result['data']<1).all() else [[x[0]+left, x[1]+top] for x in gen_result['data']]
                cls_gen.append(gen_result['data'])
                cls_score.append(gen_result['confidence_level'])
                attrs_scores.append(gen_result['attrs_scores'])
        num_res = len(cls_gen)
        if num_res>0:
            cls_score = np.stack(cls_score)
            attrs_scores = np.stack(attrs_scores)
            
            cls_gens[clsname]['data'] = cls_gen
            cls_gens[clsname]['confidence_level'] = cls_score
            # import pdb;pdb.set_trace()
            cls_gens[clsname]['attrs_scores'] = cls_score*attrs_scores
        else:
            cls_gens[clsname]['data'] = []
            cls_gens[clsname]['confidence_level'] = []
            cls_gens[clsname]['attrs_scores'] = []
    return cls_gens

def gen_distinct_instances(gen_lines,
                           gt_lines,
                           line_width=1.,
                           distance_threshold=2,
                           metric='chamfer'):
    if metric == 'chamfer':
        if distance_threshold >0:
            distance_threshold= -distance_threshold

    num_gens = gen_lines['data'].shape[0]
    num_gts = gt_lines['data'].shape[0]
    gen_scores = gen_lines['confidence_level'] # n
    
    # distance matrix: n x m
    matrix = custom_polyline_score_distinct(
            gen_lines['data'].reshape(num_gens,-1,2), 
            gt_lines['data'].reshape(num_gts,-1,2),
            linewidth=line_width,
            metric=metric)

    aggregation_index_0, aggregation_index_1 = np.where(matrix>distance_threshold)
    delete_indexes = []
    if not len(aggregation_index_0):
        return gen_lines
    for i in range(len(aggregation_index_0)):
        if gen_scores[aggregation_index_0[i]] > gen_scores[aggregation_index_1[i]]:
            delete_index = aggregation_index_1[i]
        else:
            delete_index = aggregation_index_0[i]
        if delete_index not in delete_indexes:
            delete_indexes.append(delete_index)
    if len(delete_indexes):
        gen_lines['data'] = np.delete(gen_lines['data'], delete_indexes, 0)
        gen_lines['confidence_level'] = np.delete(gen_lines['confidence_level'], delete_indexes, 0)
        gen_lines['attrs_scores'] = np.delete(gen_lines['attrs_scores'], delete_indexes, 0)

    return gen_lines

def custom_polyline_score_distinct(pred_lines, 
                                   gt_lines, 
                                   linewidth=1., 
                                   metric='chamfer'):

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_line_type_with_conf(
    attr_score, 
    lane_type=['solid', 'dotted', 'solid_fishbone', 'ldotted_fishbone', 'unknown'],
    conf_thresh=0.3
):
    # hard-code
    ltype_score = softmax(attr_score[3:8])
    index = np.argmax(ltype_score)
    if ltype_score[index] >= conf_thresh:
        ltype = lane_type[index]
        return ltype
    return "unknown"

def check_sametype_line(type1, type2):
    if "solid" in type1 and "dotted" in type2:
        return False
    if "dotted" in type1 and "solid" in type2:
        return False
    return True

def gen_aggregated_instances_with_attr(aggregated_lines,
                                       unaggregated_lines,
                                       aggregated_lines_scores,
                                       unaggregated_lines_scores,
                                       aggregated_lines_attrs_scores, 
                                       unaggregated_lines_attrs_scores, 
                                       line_width=2.0,
                                       distance_threshold=0.5,
                                       length_threshold=30,
                                       metric='chamfer',
                                       image_path=None,
                                       image_idx=0,
                                       clsname="lane",):   
    if metric == 'iou':
        line_width = 1.
    while len(unaggregated_lines):
        aggregated_lines_shapely = \
            [LineString(i).buffer(line_width,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                            for i in aggregated_lines]
        unaggregated_lines_shapely =\
            [LineString(i).buffer(line_width,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                            for i in unaggregated_lines]
        # construct tree
        index_by_id_unaggregated = dict((id(pt), i) for i, pt in enumerate(unaggregated_lines_shapely))
        index_by_id_aggregated = dict((id(pt), i) for i, pt in enumerate(aggregated_lines_shapely))
        pop_flag = False
        o = unaggregated_lines_shapely[0]
        unaggregated_id = index_by_id_unaggregated[id(o)]
        unaggre_lane_type_withconf = get_line_type_with_conf(unaggregated_lines_attrs_scores[unaggregated_id])
        for i, pline in enumerate(aggregated_lines_shapely):
            aggregated_id = index_by_id_aggregated[id(pline)]
            if clsname == "lane":
                aggre_lane_type_withconf = get_line_type_with_conf(aggregated_lines_attrs_scores[aggregated_id])
                if not check_sametype_line(aggre_lane_type_withconf, unaggre_lane_type_withconf):
                    continue
            if o.intersects(pline):
                union = [(unaggregated_lines[unaggregated_id], aggregated_lines[aggregated_id]),\
                         (np.flip(unaggregated_lines[unaggregated_id],axis=0), aggregated_lines[aggregated_id])]
                count_flag=0
                for unaggregated_line, aggregated_line in union:
                    dist_mat = distance.cdist(unaggregated_line, aggregated_line, 'euclidean')
                    unaggregated_2_aggregated = dist_mat.min(1)
                    aggregated_2_unaggregated = dist_mat.min(0)
                    unaggregated_2_aggregated_indexes= np.where(unaggregated_2_aggregated<=distance_threshold)[0]
                    aggregated_2_unaggregated_indexes= np.where(aggregated_2_unaggregated<=distance_threshold)[0]
                    if len(unaggregated_2_aggregated_indexes) >= length_threshold:
                        count_flag+=1
                        if set(range(0, 5)).intersection(set(unaggregated_2_aggregated_indexes)) \
                            and set(range(len(aggregated_line)-60, len(aggregated_line))).intersection(set(aggregated_2_unaggregated_indexes)):
                            if unaggregated_2_aggregated_indexes[-1]+1 != len(unaggregated_line) and dist_mat[unaggregated_2_aggregated_indexes[-1],aggregated_2_unaggregated_indexes[-1]]<=distance_threshold:
                                aggregated_lines[aggregated_id] = np.concatenate([aggregated_line[:aggregated_2_unaggregated_indexes[-1]], unaggregated_line[unaggregated_2_aggregated_indexes[-1]:]], 0)
                                aggregated_lines_scores[aggregated_id] = (unaggregated_lines_scores[unaggregated_id]+aggregated_lines_scores[aggregated_id])/2
                                aggregated_lines_attrs_scores[aggregated_id] +=unaggregated_lines_attrs_scores[unaggregated_id]
                            unaggregated_lines.pop(unaggregated_id)
                            unaggregated_lines_scores = np.delete(unaggregated_lines_scores, unaggregated_id, 0)
                            unaggregated_lines_attrs_scores = np.delete(unaggregated_lines_attrs_scores, unaggregated_id, 0)
                            pop_flag = True
                            break
                        elif set(range(0, 60)).intersection(set(aggregated_2_unaggregated_indexes)) \
                            and set(range(len(unaggregated_line)-5, len(unaggregated_line))).intersection(set(unaggregated_2_aggregated_indexes)):
                            if aggregated_2_unaggregated_indexes[-1]+1 != len(aggregated_line) and dist_mat[unaggregated_2_aggregated_indexes[-1],aggregated_2_unaggregated_indexes[-1]]<=distance_threshold:
                                aggregated_lines[aggregated_id] = np.concatenate([unaggregated_line[:unaggregated_2_aggregated_indexes[-1]], aggregated_line[aggregated_2_unaggregated_indexes[-1]:]], 0)
                                aggregated_lines_scores[aggregated_id] = (unaggregated_lines_scores[unaggregated_id]+aggregated_lines_scores[aggregated_id])/2
                                aggregated_lines_attrs_scores[aggregated_id] +=unaggregated_lines_attrs_scores[unaggregated_id]
                            elif aggregated_2_unaggregated_indexes[-1]+1 == len(aggregated_line) and dist_mat[unaggregated_2_aggregated_indexes[-1],aggregated_2_unaggregated_indexes[-1]]<=distance_threshold:
                                aggregated_lines[aggregated_id] = unaggregated_line
                                aggregated_lines_scores[aggregated_id] = unaggregated_lines_scores[unaggregated_id]
                                aggregated_lines_attrs_scores[aggregated_id] =unaggregated_lines_attrs_scores[unaggregated_id]
                                break
                            unaggregated_lines.pop(unaggregated_id)
                            unaggregated_lines_scores = np.delete(unaggregated_lines_scores, unaggregated_id, 0)
                            unaggregated_lines_attrs_scores = np.delete(unaggregated_lines_attrs_scores, unaggregated_id, 0)
                            pop_flag = True
                            break
                    if count_flag == 2:
                        unaggregated_lines.pop(unaggregated_id)
                        unaggregated_lines_scores = np.delete(unaggregated_lines_scores, unaggregated_id, 0)
                        unaggregated_lines_attrs_scores = np.delete(unaggregated_lines_attrs_scores, unaggregated_id, 0)
                        pop_flag=True
                if pop_flag:
                    break
        if aggregated_id == len(aggregated_lines_shapely) - 1 and not pop_flag:
            aggregated_lines.append(unaggregated_lines.pop(unaggregated_id))
            aggregated_lines_scores = np.concatenate([aggregated_lines_scores, unaggregated_lines_scores[unaggregated_id][np.newaxis,:]],  0)
            aggregated_lines_attrs_scores = np.concatenate([aggregated_lines_attrs_scores, unaggregated_lines_attrs_scores[unaggregated_id][np.newaxis,:]],  0)
            unaggregated_lines_scores = np.delete(unaggregated_lines_scores, unaggregated_id, 0)
            unaggregated_lines_attrs_scores = np.delete(unaggregated_lines_attrs_scores, unaggregated_id, 0)
    return {'data':aggregated_lines, 'confidence_level':aggregated_lines_scores, 'attrs_scores':aggregated_lines_attrs_scores}
    
def remove_dot(processed_lines,
                unprocessed_lines,
                processed_lines_scores,
                unprocessed_lines_scores,
                processed_lines_attrs_scores, 
                unprocessed_lines_attrs_scores,
                line_width=2.0,
                distance_threshold=0.5,
                length_threshold=30,
                metric='chamfer',
                image_path=None):
    while len(unprocessed_lines):
        processed_lines_shapely = \
            [LineString(i).buffer(line_width,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                            for i in processed_lines]
        unprocessed_lines_shapely =\
            [LineString(i).buffer(line_width,
                cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                            for i in unprocessed_lines]
        # construct tree
        index_by_id_processed = dict((id(pt), i) for i, pt in enumerate(processed_lines_shapely))
        index_by_id_unprocessed = dict((id(pt), i) for i, pt in enumerate(unprocessed_lines_shapely))
        pop_flag = False
        o = unprocessed_lines_shapely[0]
        unprocessed_id = index_by_id_unprocessed[id(o)]
        
        for i, pline in enumerate(processed_lines_shapely):
            processed_id = index_by_id_processed[id(pline)]
            if o.intersects(pline):
                if np.argmax(processed_lines_attrs_scores[processed_id][3:8]) != np.argmax(unprocessed_lines_attrs_scores[unprocessed_id][3:8]):
                    processed_line = processed_lines[processed_id]
                    unprocessed_line = unprocessed_lines[unprocessed_id]
                    dist_mat = distance.cdist(unprocessed_line, processed_line, 'euclidean')
                    if np.argmax(processed_lines_attrs_scores[processed_id][3:8]) == 1:
                        processed_2_unprocessed = dist_mat.min(0)
                        processed_2_unprocessed_indexes= np.where(processed_2_unprocessed<=distance_threshold)[0]
                        processed_lines[processed_id] = np.delete(processed_lines[processed_id], processed_2_unprocessed_indexes, 0)
                        if len(processed_lines[processed_id]) < 2:
                            processed_lines.pop(processed_id)
                            processed_lines_scores = np.delete(processed_lines_scores, processed_id, 0)
                            processed_lines_attrs_scores = np.delete(processed_lines_attrs_scores, processed_id, 0)
                            break
                    elif np.argmax(unprocessed_lines_attrs_scores[unprocessed_id][3:8]) == 1:
                        unprocessed_2_processed = dist_mat.min(1)
                        unprocessed_2_processed_indexes= np.where(unprocessed_2_processed<=distance_threshold)[0]
                        unprocessed_lines[unprocessed_id] = np.delete(unprocessed_lines[unprocessed_id], unprocessed_2_processed_indexes, 0)
                        if len(unprocessed_lines[unprocessed_id]) < 2:
                            unprocessed_lines.pop(unprocessed_id)
                            unprocessed_lines_scores = np.delete(unprocessed_lines_scores, unprocessed_id, 0)
                            unprocessed_lines_attrs_scores = np.delete(unprocessed_lines_attrs_scores, unprocessed_id, 0)
                            break
                    processed_lines.append(unprocessed_lines.pop(unprocessed_id))
                    processed_lines_scores = np.concatenate([processed_lines_scores, unprocessed_lines_scores[unprocessed_id][np.newaxis,:]],  0)
                    processed_lines_attrs_scores = np.concatenate([processed_lines_attrs_scores, unprocessed_lines_attrs_scores[unprocessed_id][np.newaxis,:]],  0)
                    unprocessed_lines_scores = np.delete(unprocessed_lines_scores, unprocessed_id, 0)
                    unprocessed_lines_attrs_scores = np.delete(unprocessed_lines_attrs_scores, unprocessed_id, 0)
                    pop_flag = True
                    break
        if processed_id == len(processed_lines_shapely) - 1 and not pop_flag:
            processed_lines.append(unprocessed_lines.pop(unprocessed_id))
            processed_lines_scores = np.concatenate([processed_lines_scores, unprocessed_lines_scores[unprocessed_id][np.newaxis,:]],  0)
            processed_lines_attrs_scores = np.concatenate([processed_lines_attrs_scores, unprocessed_lines_attrs_scores[unprocessed_id][np.newaxis,:]],  0)
            unprocessed_lines_scores = np.delete(unprocessed_lines_scores, unprocessed_id, 0)
            unprocessed_lines_attrs_scores = np.delete(unprocessed_lines_attrs_scores, unprocessed_id, 0)

    return {'data':processed_lines, 'confidence_level':processed_lines_scores, 'attrs_scores':processed_lines_attrs_scores}

