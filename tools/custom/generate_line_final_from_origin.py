import json
from tqdm import tqdm
import os
import cv2
from shapely.geometry import Polygon, LineString, GeometryCollection, MultiLineString, box, MultiPolygon
import numpy as np
import multiprocessing
from functools import partial
from collections import defaultdict
from argparse import ArgumentParser
import shutil
import mmcv

def generate_line_final_from_original(source_file_path, data_root, out_dir, visualize_flag):
    with open(source_file_path, 'r') as f:
        json_list = json.load(f)
    cropped_save_dir = os.path.join(out_dir, 'trajectory_cropped_images')
    if not os.path.exists(cropped_save_dir):
        os.makedirs(cropped_save_dir)
    # results = [process_line_one_single_image(image_ann, cropped_save_dir) for image_ann in tqdm(json_list)]
    with multiprocessing.Pool(processes = 4) as pool:
        partial_func = partial(process_line_one_single_image, data_root=data_root, save_dir=cropped_save_dir)
        results = list(tqdm(pool.imap(partial_func, json_list), total=len(json_list), desc="process every 6k image to extract the 1k gt"))
    result_dict = {}
    for result in tqdm(results, desc='format dict'):
        if result==None:
            continue
        elif len(result[1]) > 0:
            result_dict[result[0]] = result[1]
    
    final_dict = defaultdict(list)
    big_image_name_list = list(result_dict.keys())[:]
    
    length = len(big_image_name_list)
    train_data_length = round(0.8*length)
    train_data_name_list = big_image_name_list[:train_data_length]
    valid_data_name_list = big_image_name_list[train_data_length:]
    # with open(os.path.join(out_dir, 'train_valid_6k_image_name.json'), 'w') as f:
    #     json.dump(train_valid_6k_image_name, f)
    for big_image_name in big_image_name_list:
        if big_image_name in train_data_name_list:
            final_dict['train'].extend(result_dict[big_image_name])
        else:
            final_dict['valid'].extend(result_dict[big_image_name])
    result_1k_gt_dict_json_path = os.path.join(out_dir, 'sd_data_line_dict.json')
    with open(result_1k_gt_dict_json_path, 'w') as f:
        json.dump(final_dict, f)
    print('generate line completed')
    if visualize_flag:
        visualize_dir = '/'.join(cropped_save_dir.split('/')[:-1]+['visualize_1k_gt'])
        print('the 1k gt will be visualize in {}'.format(visualize_dir))
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        for small_image_annotations in tqdm(result_dict.values(), desc='visualize 1k gt'):
            for small_image_annotation in small_image_annotations:
                visualize_small_image_line(small_image_annotation, cropped_save_dir, visualize_dir)
        print('visualization complete')

def process_line_one_single_image(image_ann, data_root, save_dir):
    image_name = image_ann['image_key']
    image_path = os.path.join(data_root, 'image_data', image_name) + '.jpg'
    trajectory_json_path = image_path.replace('image_data', 'trajectory_data').replace('.jpg', '.json')
    instances_list = extract_line_useful_data(image_ann)
    trajectory_data_list = extract_trajectory_data(trajectory_json_path)
    small_image_data_list = extract_line_1k_small_image_instances(trajectory_data_list, instances_list, image_name, image_path, save_dir)
    return image_name, small_image_data_list

def extract_line_useful_data(img_ann):
    instances_list = []
    if 'parsing' not in img_ann.keys():
        for key, value in img_ann.items():
            if key == 'curb':
                for instance in value:
                    new_instance_ann = {}
                    new_instance_attr = {}
                    new_instance_ann['class'] = 'curb'
                    new_instance_ann['data'] = instance['data']
                    instance_attr = instance['attrs'].pop('attribution')
                    if instance_attr in ['no', 'unknown', 'other']:
                        new_instance_attr['curb_type'] = 'unknown'
                    else:
                        new_instance_attr['curb_type'] = instance_attr
                    new_instance_ann['attrs'] = new_instance_attr
                    instances_list.append(new_instance_ann.copy())
            elif key == 'stop_line':
                for instance in value:
                    new_instance_ann = {}
                    new_instance_ann['class'] = 'stopline'
                    new_instance_ann['attrs'] = {}
                    if 'segments' in instance.keys():
                        new_instance_ann['data'] = [x['data'] for x in instance['segments'][0]['points']]
                    else:
                        new_instance_ann['data'] = instance['data']
                    instances_list.append(new_instance_ann.copy())
            elif key == 'lane':
                for instance in value: 
                    new_instance_ann = {}
                    new_instance_ann['class'] = 'lane'
                    if 'segments' in instance.keys():
                        new_instance_ann['data'] = [x['data'] for x in instance['segments'][0]['points']]
                    else:
                        new_instance_ann['data'] = instance['data']
                    new_instance_attr = {}
                    if 'attrs' in instance.keys():
                        new_instance_attr['lane_direction'] = instance['attrs']['direction']
                        if instance['attrs']['type'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_type'] = 'unknown'
                        else:
                            new_instance_attr['lane_type'] = instance['attrs']['type']
                        if instance['attrs']['properties'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_properties'] = 'unknown'
                        else:
                            new_instance_attr['lane_properties'] = instance['attrs']['properties']
                        if instance['attrs']['color'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_color'] = 'unknown'
                        else:
                            new_instance_attr['lane_color'] = instance['attrs']['color']
                        if instance['attrs']['flag'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_flag'] = 'unknown'
                        else:
                            new_instance_attr['lane_flag'] = instance['attrs']['flag']
                        if instance['attrs']['width'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_width'] = 'unknown'
                        else:
                            new_instance_attr['lane_width'] = instance['attrs']['width']
                        new_instance_ann['attrs'] = new_instance_attr
                    elif 'attr' in instance.keys():
                        new_instance_attr['lane_direction'] = instance['attr']['direction']
                        if instance['attr']['type'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_type'] = 'unknown'
                        else:
                            new_instance_attr['lane_type'] = instance['attr']['type']
                        if instance['attr']['properties'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_properties'] = 'unknown'
                        else:
                            new_instance_attr['lane_properties'] = instance['attr']['properties']
                        if instance['attr']['color'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_color'] = 'unknown'
                        else:
                            new_instance_attr['lane_color'] = instance['attr']['color']
                        if instance['attr']['flag'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_flag'] = 'unknown'
                        else:
                            new_instance_attr['lane_flag'] = instance['attr']['flag']
                        if instance['attr']['width'] in ['no', 'unknown', 'other']:
                            new_instance_attr['lane_width'] = 'unknown'
                        else:
                            new_instance_attr['lane_width'] = instance['attr']['width']
                        new_instance_ann['attrs'] = new_instance_attr
                    # else:
                    #     print(list(instance.keys()))
                    instances_list.append(new_instance_ann.copy())
            # elif key == 'ignore':
            #     # import pdb;pdb.set_trace()
            #     for instance in value: 
            #         new_instance_ann = {}
            #         new_instance_ann['class'] = 'ignore'
            #         if 'segments' in instance.keys():
            #             new_instance_ann['data'] = [x['data'] for x in instance['segments'][0]['points']]
            #         else:
            #             new_instance_ann['data'] = instance['data']
            #         new_instance_ann['attrs'] = {}
            #         instances_list.append(new_instance_ann.copy())
            # else:
            #     print(key)
    else:
        for instance in img_ann['parsing']:
            new_instance_ann = {}
            new_instance_attr = {}
            if instance['attrs']['type'] == 'lane_marking':
                new_instance_ann['class'] = 'lane'
                new_instance_ann['data'] = instance['data']
                new_instance_attr['lane_direction'] = 'unidirectional'
                if instance['attrs']['lane_type'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_type'] = 'unknown'
                else:
                    new_instance_attr['lane_type'] = instance['attrs']['lane_type']
                if instance['attrs']['lane_properties'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_properties'] = 'unknown'
                else:
                    new_instance_attr['lane_properties'] = instance['attrs']['lane_properties']
                if instance['attrs']['lane_color'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_color'] = 'unknown'
                else:
                    new_instance_attr['lane_color'] = instance['attrs']['lane_color']
                if instance['attrs']['flag'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_flag'] = 'unknown'
                else:
                    new_instance_attr['lane_flag'] = instance['attrs']['flag']
                if instance['attrs']['width'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_width'] = 'unknown'
                else:
                    new_instance_attr['lane_width'] = instance['attrs']['width']
                new_instance_ann['attrs'] = new_instance_attr
                instances_list.append(new_instance_ann.copy())
            elif instance['attrs']['type'] == 'bidirectional_lane':
                new_instance_ann['class'] = 'lane'
                new_instance_ann['data'] = instance['data']
                new_instance_attr['lane_direction'] = 'unidirectional'
                if instance['attrs']['lane_type'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_type'] = 'unknown'
                else:
                    new_instance_attr['lane_type'] = instance['attrs']['lane_type']
                if instance['attrs']['lane_properties'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_properties'] = 'unknown'
                else:
                    new_instance_attr['lane_properties'] = instance['attrs']['lane_properties']
                if instance['attrs']['lane_color'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_color'] = 'unknown'
                else:
                    new_instance_attr['lane_color'] = instance['attrs']['lane_color']
                if instance['attrs']['flag'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_flag'] = 'unknown'
                else:
                    new_instance_attr['lane_flag'] = instance['attrs']['flag']
                if instance['attrs']['width'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_width'] = 'unknown'
                else:
                    new_instance_attr['lane_width'] = instance['attrs']['width']
                new_instance_ann['attrs'] = new_instance_attr
                instances_list.append(new_instance_ann.copy())
            elif instance['attrs']['type'] == 'lane_unknown':
                new_instance_ann['class'] = 'lane'
                new_instance_ann['data'] = instance['data']
                new_instance_attr['lane_direction'] = 'unknown'
                if instance['attrs']['lane_type'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_type'] = 'unknown'
                else:
                    new_instance_attr['lane_type'] = instance['attrs']['lane_type']
                if instance['attrs']['lane_properties'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_properties'] = 'unknown'
                else:
                    new_instance_attr['lane_properties'] = instance['attrs']['lane_properties']
                if instance['attrs']['lane_color'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_color'] = 'unknown'
                else:
                    new_instance_attr['lane_color'] = instance['attrs']['lane_color']
                if instance['attrs']['flag'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_flag'] = 'unknown'
                else:
                    new_instance_attr['lane_flag'] = instance['attrs']['flag']
                if instance['attrs']['width'] in ['no', 'unknown', 'other']:
                    new_instance_attr['lane_width'] = 'unknown'
                else:
                    new_instance_attr['lane_width'] = instance['attrs']['width']
                new_instance_ann['attrs'] = new_instance_attr
                instances_list.append(new_instance_ann.copy())
            elif instance['attrs']['type'] == 'Road_teeth':
                new_instance_ann['class'] = 'curb'
                new_instance_ann['data'] = instance['data']
                if instance['attrs']['curb'] in ['no', 'unknown', 'other']:
                    new_instance_attr['curb_type'] = 'unknown'
                else:
                    new_instance_attr['curb_type'] = instance['attrs']['curb']
                new_instance_ann['attrs'] = new_instance_attr
                instances_list.append(new_instance_ann.copy())
            elif instance['attrs']['type'] == 'Stop_Line':
                new_instance_ann['class'] = 'stopline'
                new_instance_ann['data'] = instance['data']
                new_instance_ann['attrs'] = {}
                instances_list.append(new_instance_ann.copy())
            # else:
            #     print(instance['attrs']['type'])
            # elif instance['attrs']['type'] == 'ignore':
            #     new_instance_ann['class'] = 'ignore'
            #     new_instance_ann['data'] = instance['data']
            #     new_instance_ann['attrs'] = new_instance_attr
            #     instances_list.append(new_instance_ann.copy())
    return instances_list

def extract_trajectory_data(trajectory_json_path):
    return mmcv.load(trajectory_json_path)

def extract_line_1k_small_image_instances(trajectory_data_list, instances_list, image_name, image_path, save_dir):
    big_image = cv2.imread(image_path)
    i = 0
    small_instance_data_list = []

    for trajectory_data in trajectory_data_list:
        if len(trajectory_data) < 2:
            continue
        trajectory_shapely = LineString(np.array(trajectory_data))
        sample_num = 7
        distances = np.linspace(0, trajectory_shapely.length, sample_num)
        sampled_points = np.array([list(trajectory_shapely.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        for sampled_point in sampled_points:
            x, y = sampled_point
            if x < 500:
                x = 500
            elif x > 5500:
                x = 5500
            if y < 500:
                y = 500
            elif y > 5500:
                y = 5500
            bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = int(round(x) - 500), int(round(y) - 500), int(round(x) + 499), int(round(y) + 499)
            cropped = big_image[bbox_miny:bbox_maxy+1, bbox_minx:bbox_maxx+1]
            assert cropped.shape[0] == cropped.shape[1], print(cropped.shape)
            small_image_name = '_'.join([image_name, str(i)]) + '.jpg'

            bounding_box = box(bbox_minx, bbox_miny, bbox_maxx, bbox_maxy) 
            small_instances_data = []
            small_image_dict = {}
            i += 1
            # crop
            for instance in instances_list:
                instance_cls = instance['class']
                instance_pts = instance['data']
                instance_attr = instance['attrs']
                new_instance_pts = []
                for coord in instance_pts:
                    if len(coord) != 2:
                        coord = coord[:2]
                    x = float(coord[0])
                    y = float(coord[1])
                    new_instance_pts.append([x, y])
                if len(new_instance_pts) == 1:
                    continue
                if new_instance_pts[0] == new_instance_pts[-1]:
                    if len(new_instance_pts) == 2:
                        continue
                    instance_poly = Polygon(np.array(new_instance_pts)).buffer(0.001)
                else:
                    instance_poly = LineString(np.array(new_instance_pts))
                line = instance_poly.intersection(bounding_box)
                if isinstance(line, MultiLineString):
                    for one in line.geoms:
                        new_coords = [[x[0]-bbox_minx, x[1]-bbox_miny] for x in list(one.coords)]
                        small_instances_data.append({'class':instance_cls, 'data':new_coords, 'attrs':instance_attr})
                elif isinstance(line, LineString):
                    if len(list(line.coords))!=0:
                        new_coords = [[x[0]-bbox_minx, x[1]-bbox_miny] for x in list(line.coords)]
                        small_instances_data.append({'class':instance_cls, 'data':new_coords, 'attrs':instance_attr})
                elif isinstance(line, GeometryCollection):
                    for one in line.geoms:
                        if isinstance(one, LineString):
                            if len(list(one.coords))!=0:
                                new_coords = [[x[0]-bbox_minx, x[1]-bbox_miny] for x in list(one.coords)]
                                small_instances_data.append({'class':instance_cls, 'data':new_coords, 'attrs':instance_attr})
                        # else:
                        #     print(one)
                elif isinstance(line, Polygon):
                    if len(list(line.exterior.coords))!=0:
                        new_coords = [[x[0]-bbox_minx, x[1]-bbox_miny] for x in list(line.exterior.coords)[:-1]]
                        small_instances_data.append({'class':instance_cls, 'data':new_coords, 'attrs':instance_attr})
                elif isinstance(line, MultiPolygon):
                    for one in line.geoms:
                        if len(list(one.exterior.coords))!=0:
                            new_coords = [[x[0]-bbox_minx, x[1]-bbox_miny] for x in list(one.exterior.coords)[:-1]]
                            small_instances_data.append({'class':instance_cls, 'data':new_coords, 'attrs':instance_attr})
                # else:
                #     print(line)
            if len(small_instances_data) == 0:
                continue
            else:
                small_image_dict['image_name'] = small_image_name
                small_image_dict['left_top'] = [bbox_minx, bbox_miny]
                small_image_dict['instances'] = small_instances_data
                small_instance_data_list.append(small_image_dict)
                cv2.imwrite(os.path.join(save_dir, small_image_name), cropped)
    return small_instance_data_list

def visualize_small_image_line(image_ann, data_root, out_dir):
    image_name = image_ann['image_name']
    image_path = os.path.join(data_root, image_name)
    gt_instances = image_ann['instances']
    sampled_gt_instances = gt_instances
    image = cv2.imread(image_path)
    
    color = {'lane':(0,0,0),'curb':(0, 0, 255), 'stopline':(0,255,255), 'ignore':(0,255,255), 'trajectory':(0,255,0)}
    for idx in range(len(sampled_gt_instances)):
        sub_gt_points = sampled_gt_instances[idx]['data']
        sub_gt_points = [tuple([round(float(x[0])), round(float(x[1]))]) for x in sub_gt_points]
        instance_class = sampled_gt_instances[idx]['class']
        color_fill = color[instance_class]
        line_width = 8 if instance_class == 'ignore' else 2
        for i in range(len(sub_gt_points)): 
            if i == 0:
                cv2.circle(image, sub_gt_points[i], 15, (0,97,255), -1)
                continue
            elif i == len(sub_gt_points)-1:
                cv2.line(image, sub_gt_points[i-1], sub_gt_points[i], color_fill, 4)
                cv2.circle(image, sub_gt_points[i], 10, (0,255,255), -1)
            else:
                cv2.line(image, sub_gt_points[i-1], sub_gt_points[i], color_fill, 4)
                cv2.circle(image, sub_gt_points[i], 8, (0,255,0), -1)
    drew_image = cv2.copyMakeBorder(image, 0, 0, 50, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
    original_image = cv2.imread(image_path)
    concat_image = cv2.hconcat([original_image, drew_image])
    drew_image_path = out_dir + '/' + image_name
    cv2.imwrite(drew_image_path, concat_image)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('all_gt_6k_json_file_path', help='Input all 6k gt json file')
    parser.add_argument('data_root', help='The directory to store origin 6k image and trajectory data')
    parser.add_argument('out_dir', help='The directory to save cropped image')
    parser.add_argument("--visualize", action='store_true', help='Whether to visualize the 1k gt')
    args = parser.parse_args()

    generate_line_final_from_original(args.all_gt_6k_json_file_path, args.data_root, args.out_dir, args.visualize)