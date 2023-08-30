import scipy
import numpy as np
from os import path as osp
from PIL import Image
import torch
from tqdm import tqdm
from  multiprocessing import Pool
from functools import partial
from sklearn import metrics
import warnings
import json
warnings.filterwarnings("ignore")
from shapely.geometry import LineString
from .mean_ap import average_precision
from .tpfp import tpfp_gen, custom_tpfp_gen_1
import mmcv
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from mmcv.utils import print_log
from terminaltables import AsciiTable
from collections import defaultdict
from functools import partial
import multiprocessing

def get_cls_results(gen_results, 
                    annotations,
                    num_sample=100, 
                    num_pred_pts_per_instance=30,
                    eval_use_same_gt_sample_num_flag=False,
                    class_id=0, 
                    fix_interval=False):
    """Get det results and gt information of a certain class.

    Args:
        gen_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes
    """
    # if len(gen_results) == 0 or 
    # import pdb;pdb.set_trace()
    # print(len(gen_results))
    # print(len(annotations))
    cls_gens, cls_scores, gen_attrs = [], [], []
    for res in gen_results:
        if res['type'] == class_id:
            if len(res['pts']) < 2:
                continue
            if not eval_use_same_gt_sample_num_flag:
                sampled_points = np.array(res['pts'])
            else:
                line = res['pts']
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
            cls_gens.append(sampled_points)
            cls_scores.append(res['confidence_level'])
            gen_attrs.append(res['attrs'])

    num_res = len(cls_gens)
    if num_res > 0:
        cls_gens = np.stack(cls_gens).reshape(num_res,-1)
        cls_scores = np.array(cls_scores)[:,np.newaxis]
        cls_gens = np.concatenate([cls_gens,cls_scores],axis=-1)
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')
    else:
        if not eval_use_same_gt_sample_num_flag:
            cls_gens = np.zeros((0,num_pred_pts_per_instance*2+1))
        else:
            cls_gens = np.zeros((0,num_sample*2+1))
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')

    cls_gts, gt_attrs = [], []
    for ann in annotations:
        if ann['type'] == class_id:
            # line = ann['pts'] +  np.array((1,1)) # for hdmapnet
            line = ann['pts']
            # line = ann['pts'].cumsum(0)
            line = LineString(line)
            distances = np.linspace(0, line.length, num_sample)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                        for distance in distances]).reshape(-1, 2)
            
            cls_gts.append(sampled_points)
            gt_attrs.append(ann['attrs'])
    num_gts = len(cls_gts)
    if num_gts > 0:
        cls_gts = np.stack(cls_gts).reshape(num_gts,-1)
    else:
        cls_gts = np.zeros((0,num_sample*2))
    return cls_gens, cls_gts, gen_attrs, gt_attrs
    # ones = np.ones((num_gts,1))
    # tmp_cls_gens = np.concatenate([cls_gts,ones],axis=-1)
    # return tmp_cls_gens, cls_gts

def format_res_gt_by_classes(result_path,
                             gen_results,
                             annotations,
                             cls_names=None,
                             num_pred_pts_per_instance=30,
                             eval_use_same_gt_sample_num_flag=False,
                             nproc=24):
    assert cls_names is not None
    timer = mmcv.Timer()
    fix_interval = False
    print('results path: {}'.format(result_path))
    
    output_dir = osp.join(*osp.split(result_path)[:-1])
    # import pdb;pdb.set_trace()
    assert len(gen_results) == len(annotations)

    pool = Pool(nproc)
    cls_gens, cls_gts = {}, {}
    print('Formatting ...')
    formatting_file = 'cls_formatted.pkl'
    formatting_file = osp.join(output_dir,formatting_file)
    # arranged_pred_results, arranged_gt_annotations = arrange_results_and_annotations(gen_results, annotations)
    num_fixed_sample_pts = 100
    for i, clsname in enumerate(cls_names):
        # import pdb;pdb.set_trace()
        # for gen_result, annotation in zip(gen_results, annotations):
        #     gengts = get_cls_results(gen_result,
        #                     annotation,
        #                     num_sample=num_fixed_sample_pts,
        #                     num_pred_pts_per_instance=num_pred_pts_per_instance,
        #                     eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        #                     class_id=i,
        #                     fix_interval=fix_interval)
            # gens, gts, gens_attrs, gts_attrs = tuple(zip(*gengts))
        gengts = pool.starmap(
                partial(get_cls_results, num_sample=num_fixed_sample_pts,
                    num_pred_pts_per_instance=num_pred_pts_per_instance,
                    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
                    class_id=i,
                    fix_interval=fix_interval),
                zip(gen_results, annotations))   

        gens, gts, gens_attrs, gts_attrs = tuple(zip(*gengts))    
        cls_gens[clsname] = {'data':gens, 'attrs':gens_attrs}
        cls_gts[clsname] = {'data':gts, 'attrs':gts_attrs}
    # import pdb;pdb.set_trace()
    mmcv.dump([cls_gens, cls_gts],formatting_file)
    print('Cls data formatting done in {:2f}s!! with {}'.format(float(timer.since_start()),formatting_file))
    pool.close()
    return cls_gens, cls_gts

def eval_map(gen_results,
             annotations,
             cls_gens,
             cls_gts,
             threshold=0.5,
             cls_names=None,
             attrs_dict=None,
             logger=None,
             tpfp_fn=None,
             metric=None,
             num_pred_pts_per_instance=30,
             nproc=24):
    timer = mmcv.Timer()
    pool = Pool(nproc)

    eval_results = []
    
    for i, clsname in enumerate(cls_names):
        # import pdb;pdb.set_trace()
        # get gt and det bboxes of this class
        cls_gen_data = cls_gens[clsname]['data']
        cls_gt_data = cls_gts[clsname]['data']

        cls_gen_attrs = cls_gens[clsname]['attrs']
        cls_gt_attrs = cls_gts[clsname]['attrs']
        # choose proper function according to datasets to compute tp and fp
        # XXX
        # func_name = cls2func[clsname]
        # tpfp_fn = tpfp_fn_dict[tpfp_fn_name]
        tpfp_fn = custom_tpfp_gen_1
        # Trick for serialized
        # only top-level function can be serized
        # somehow use partitial the return function is defined
        # at the top level.
        
        # TODO this is a hack
        tpfp_fn = partial(tpfp_fn, threshold=threshold, metric=metric)
        args = []
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_fn,
            zip(cls_gen_data, cls_gt_data, *args))
        tp, fp, tp_gt, gt_covered = tuple(zip(*tpfp))
        
        num_gts = 0
        for j, bbox in enumerate(cls_gt_data):
            num_gts += bbox.shape[0]
        
        cls_gen_data = np.vstack(cls_gen_data)
        num_dets = cls_gen_data.shape[0]
        sort_inds = np.argsort(-cls_gen_data[:, -1]) #descending, high score front
        tp_sorted = np.hstack(tp)[sort_inds]
        fp_sorted = np.hstack(fp)[sort_inds]
        
        # calculate recall and precision with tp and fp
        # num_det*num_res
        tp_sorted = np.cumsum(tp_sorted, axis=0)
        fp_sorted = np.cumsum(fp_sorted, axis=0)
        eps = np.finfo(np.float32).eps
        recalls = tp_sorted / np.maximum(num_gts, eps)
        precisions = tp_sorted / np.maximum((tp_sorted + fp_sorted), eps)
        # calculate AP
        # if dataset != 'voc07' else '11points'
        mode = 'area'
        ap = average_precision(recalls, precisions, mode)
        # evaluate the attributes
        all_gt_attrs = []
        all_pred_attrs = []
        attrs_metric = dict()
        if clsname == 'lane':
            for single_gen_attrs, single_gt_attrs, gt_indice, single_gt_covered in zip(cls_gen_attrs, cls_gt_attrs, tp_gt, gt_covered):
                if len(single_gen_attrs) != 0 and len(np.array(single_gen_attrs)[gt_indice][single_gt_covered]) != 0:
                    all_pred_attrs.append(np.array(single_gen_attrs)[:,:5][gt_indice][single_gt_covered])
                    all_gt_attrs.append(np.array(single_gt_attrs)[:,:5][single_gt_covered])
            if len(all_pred_attrs):
                all_pred_attrs = np.vstack(all_pred_attrs).T
                all_gt_attrs = np.vstack(all_gt_attrs).T
                # import pdb;pdb.set_trace()
                for attr_class, single_class_pred_attrs, single_class_gt_attrs in zip(list(attrs_dict.keys())[:][:5], all_pred_attrs, all_gt_attrs):
                    single_attrs_precision = precision_score(single_class_gt_attrs, single_class_pred_attrs, average='micro')
                    single_attrs_recall = recall_score(single_class_gt_attrs, single_class_pred_attrs, average='micro')
                    single_attrs_f1 = f1_score(single_class_gt_attrs, single_class_pred_attrs, average='micro')
                    attrs_metric[attr_class] = {'precision':single_attrs_precision, 'recall':single_attrs_recall, 'f1_score':single_attrs_f1}
            else:
                for attr_class in list(attrs_dict.keys())[:][:4]:
                    attrs_metric[attr_class] = {'precision':0.0, 'recall':0.0, 'f1_score':0.0}
        if clsname == 'curb':
            for single_gen_attrs, single_gt_attrs, gt_indice, single_gt_covered in zip(cls_gen_attrs, cls_gt_attrs, tp_gt, gt_covered):
                if len(single_gen_attrs) != 0 and len(np.array(single_gen_attrs)[gt_indice][single_gt_covered]) != 0:
                    all_pred_attrs.append(np.array(single_gen_attrs)[:,-1][:,np.newaxis][gt_indice][single_gt_covered])
                    all_gt_attrs.append(np.array(single_gt_attrs)[:,-1][:,np.newaxis][single_gt_covered])
            if len(all_pred_attrs):
                # import pdb;pdb.set_trace()
                all_pred_attrs = np.vstack(all_pred_attrs).T
                all_gt_attrs = np.vstack(all_gt_attrs).T
                # import pdb;pdb.set_trace()
                for single_class_pred_attrs, single_class_gt_attrs in zip(all_pred_attrs, all_gt_attrs):
                    single_attrs_precision = precision_score(single_class_gt_attrs, single_class_pred_attrs, average='micro')
                    single_attrs_recall = recall_score(single_class_gt_attrs, single_class_pred_attrs, average='micro')
                    single_attrs_f1 = f1_score(single_class_gt_attrs, single_class_pred_attrs, average='micro')
                    # import pdb;pdb.set_trace()
                    attrs_metric[list(attrs_dict.keys())[:][-1]] = {'precision':single_attrs_precision, 'recall':single_attrs_recall, 'f1_score':single_attrs_f1}
            else:
                # import pdb;pdb.set_trace()
                attrs_metric[list(attrs_dict.keys())[:][-1]] = {'precision':0.0, 'recall':0.0, 'f1_score':0.0}
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap,
            'attrs_metric':attrs_metric
        })

        print('cls:{} done in {:2f}s!!'.format(clsname,float(timer.since_last_check())))
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if len(aps) else 0.0

    print_map_summary(
        mean_ap, eval_results, class_name=cls_names, attrs_dict= attrs_dict, logger=logger)

    return mean_ap, eval_results 

def print_map_summary(mean_ap,
                      results,
                      class_name=None,
                      attrs_dict=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return
    # import pdb;pdb.set_trace()
    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            precisions[:, i] = np.array(cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    label_names = class_name

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    class_header = ['class', 'gts', 'dets', 'precision', 'recall', 'ap']
    attrs_header = ['class', 'precision', 'recall', 'f1_score']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        class_table_data = [class_header]
        attrs_table_data = [attrs_header]
        for j in range(num_classes):
            
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{precisions[i, j]:.3f}', f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            class_table_data.append(row_data)
            # import pdb;pdb.set_trace()
            attrs_metric = results[j]['attrs_metric']
            attrs_names = list(attrs_metric.keys())[:]
            num_attrs = len(attrs_names)
            for k in range(num_attrs):
                attrs_row_data = [
                    attrs_names[k], attrs_metric[attrs_names[k]]['precision'], attrs_metric[attrs_names[k]]['recall'], attrs_metric[attrs_names[k]]['f1_score'] 
                ]
                attrs_table_data.append(attrs_row_data)
        attrs_table = AsciiTable(attrs_table_data)
        # attrs_table.inner_footing_row_border = True
        print_log('\n' + attrs_table.table, logger=logger)
        class_table_data.append(['mAP', '', '', '', '', f'{mean_ap[i]:.3f}'])
        class_table = AsciiTable(class_table_data)
        class_table.inner_footing_row_border = True
        print_log('\n' + class_table.table, logger=logger)