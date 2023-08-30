from math import sqrt
import argparse
from os import listdir
from os.path import join as opj
from os.path import exists
import os
import cv2
import math
import json
import numpy as np
import copy


def get_driving_bev_images(site_path):
    image_path = opj(site_path, "Bev_Result/Bev_img")
    if exists(image_path):
        data_list = [opj(image_path, file) for file in listdir(image_path) if file.endswith("_bev.png")]
    else:
        data_list = []
    return data_list
    
def get_parking_bev_images(pack_path):
    data_list = []
    sub_dir = opj(pack_path, "BevCrop_Result/final")
    image_key = "gridbev"
    if not exists(sub_dir):
        sub_dir = opj(pack_path, "Bev_Result/final")
        image_key = "ninegridbev"
    if exists(sub_dir):
        for i in list(range(0, 10)) + ["all"]: # up to 10 levels & all
            # /horizon-bucket/SD_Algorithm/12_perception_bev_hde/03_hde_data/park_data/UT370_20230517_D/20230517-112502_956/BevCrop_Result/final/0/Bev_img
            json_path = opj(sub_dir, str(i), "fileindexes.json")
            if exists(json_path):
                json_data = read_json(json_path)
                folder = json_data[image_key]["folder"]
                names = json_data[image_key]["names"]
                for name in names:
                    data_list.append(opj(sub_dir, str(i), folder, name))
    return data_list

def get_parking_bev_images_line(pack_path):
    data_list = []
    
    sub_dir_v12 = opj(pack_path, "BevLane_Result_v1.2/final")
    sub_dir_v11 = opj(pack_path, "BevLane_Result_v1.1/final")
    if exists(sub_dir_v12):
        for i in list(range(0, 10)) + ["all"]: # up to 10 levels & all
            # /horizon-bucket/SD_Algorithm/12_perception_bev_hde/03_hde_data/park_data/UT0F3_20230308_D/20230308-104554_937/BevLane_Result_v1.2/final/0/Bev_img
            level_i_dir = opj(sub_dir_v12, str(i), "Bev_img")
            if exists(level_i_dir):
                for file in listdir(level_i_dir):
                    if file.endswith("_bevroi.png"):
                        data_list.append(opj(level_i_dir, file))
    elif exists(sub_dir_v11):
        for i in list(range(0, 10)) + ["all"]: # up to 10 levels & all
            level_i_dir = opj(sub_dir_v11, str(i), "Bev_img")
            if exists(level_i_dir):
                for file in listdir(level_i_dir):
                    if file.endswith("_bev.png"):
                        data_list.append(opj(level_i_dir, file))
    return data_list

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def point2LineDistance(point_a, point_b, point_c):
    """  计算点a到点b c所在直线的距离  :param point_a:  :param point_b:  :param point_c:  :return:  """
    # 首先计算b c 所在直线的斜率和截距
    if point_b[0] == point_c[0]:
        return abs(point_a[0] - point_b[0])
    slope = (point_b[1] - point_c[1]) / (point_b[0] - point_c[0])
    intercept = point_b[1] - slope * point_b[0]
    
    # 计算点a到b c所在直线的距离
    distance = abs(slope * point_a[0] - point_a[1] + intercept) / sqrt(1 + pow(slope, 2))
    return distance

def point2pointDistance(point_a, point_b):
    return sqrt((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)
    
class DouglasPeuker_v3(object):
    def __init__(self,threshold_1, threshold_2):
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.qualify_list = list()
        self.disqualify_list = list()
        self.final_qualify_list = list()
 
    def diluting(self, point_list):
        """    抽稀    :param point_list:二维点列表    :return:    """
        if len(point_list) < 3:
            if len(self.qualify_list) == 0:
                self.qualify_list.extend(point_list[::-1])
            else:
                if point_list[-1] == self.qualify_list[-1]:
                    self.qualify_list.append(point_list[0])
                else:
                    self.qualify_list.extend(point_list[::-1])
        else:
        # 找到与收尾两点连线距离最大的点
            max_distance_index, max_distance = 0, 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                distance = point2LineDistance(point, point_list[0], point_list[-1])
                if distance > max_distance:
                    max_distance_index = index
                    max_distance = distance
 
            # 若最大距离小于阈值，则去掉所有中间点。 反之，则将曲线按最大距离点分割
            if max_distance < self.threshold_1:
                if len(self.qualify_list) == 0:
                    self.qualify_list.append(point_list[-1])
                    self.qualify_list.append(point_list[0])
                else:
                    if point_list[-1] == self.qualify_list[-1]:
                        self.qualify_list.append(point_list[0])
                    else:
                        self.qualify_list.append(point_list[-1])
                        self.qualify_list.append(point_list[0])
            else:
                # 将曲线按最大距离的点分割成两段
                sequence_a = point_list[:max_distance_index+1]
                sequence_b = point_list[max_distance_index:]
        
                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 3 and sequence == sequence_b:
                        if len(self.qualify_list) == 0:
                            self.qualify_list.extend(sequence[::-1])
                        else:
                            if sequence[-1] == self.qualify_list[-1]:
                                self.qualify_list.append(sequence[0])
                            else:
                                self.qualify_list.extend(sequence[::-1])
                    else:
                        self.disqualify_list.append(sequence)
 
    def main(self, point_list):
        self.diluting(point_list)
        while len(self.disqualify_list) > 0:
            self.diluting(self.disqualify_list.pop())
        length = len(self.qualify_list)
        first_index = 0
        second_index = 1
        self.final_qualify_list.append(self.qualify_list[first_index])
        while second_index<length:
            
            first_point = self.qualify_list[first_index]
            second_point = self.qualify_list[second_index]
            first_2_second_distance = point2pointDistance(first_point, second_point)
            if first_2_second_distance <= self.threshold_2:
                second_index += 1
            else:
                self.final_qualify_list.append(second_point)
                first_index = second_index
                second_index += 1
        return self.final_qualify_list

class TrajPadSampler(object):
    SIZE = 1000
    def __init__(self, imgpath, left_top):
        self.imgpath = imgpath
        self.left_top = left_top
        img = cv2.imread(self.imgpath)
        self.img = img
        canvas = np.zeros_like(img)
        for lt in self.left_top:
            cv2.rectangle(canvas ,lt, [lt[0]+self.SIZE, lt[1]+self.SIZE], (1, 1, 1), -1)
        self.canvas = canvas
    
    def get_traj_mask_ratio(self, left_top):
        x, y = left_top
        return self.canvas[y: y+self.SIZE, x: x+self.SIZE, 0].mean()

    def get_img_mask_raio(self, left_top):
        x, y = left_top
        roi = self.img[y: y+self.SIZE, x: x+self.SIZE, :]
        return (roi.sum(axis=2) > 130).mean()

    def sample(self, data_dict):
        img = data_dict['img']
        # 采样轨迹之外的区域
        infer_subimg_resolution = 1000
        bigimgsize=6000
        subimg_interval = infer_subimg_resolution // 2
        sub_img_list = []
        left_top_list = []
        
        sampled_points = []
        for offsetx in range(subimg_interval, bigimgsize+1-subimg_interval, subimg_interval):
            for offsety in range(subimg_interval, bigimgsize+1-subimg_interval, subimg_interval):
                sampled_points.append([offsetx, offsety])
        for point in sampled_points:
            x, y = point
            if x < subimg_interval:
                x = subimg_interval
            elif x > (bigimgsize - subimg_interval):
                x = (bigimgsize - subimg_interval)
            if y < subimg_interval:
                y = subimg_interval
            elif y > (bigimgsize - subimg_interval):
                y = (bigimgsize - subimg_interval)
            bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = round(x - subimg_interval), round(y - subimg_interval), \
                round(x + subimg_interval - 1), round(y + subimg_interval - 1)
            subimg = img[:, bbox_miny: bbox_maxy+1, bbox_minx: bbox_maxx+1]
            img_ratio = self.get_img_mask_raio([bbox_minx, bbox_miny])
            if img_ratio < 0.5:
                continue
            if self.get_traj_mask_ratio([bbox_minx, bbox_miny]) > 0.5:
                continue
            if not subimg.shape[-2:] == (1000, 1000):
                subimg = F.resize(subimg, (1000, 1000))
            sub_img_list.append(subimg)  # (left, upper, right, lower)
            left_top_list.append([bbox_minx, bbox_miny])
        sub_datas = []
        for i in range(len(sub_img_list)):
            sub_data_dict = dict()
            sub_data_dict['img_metas'] = data_dict['img_metas']
            sub_data_dict['img_metas'][0]['batch_input_shape'] = (sub_img_list[i].shape[1], sub_img_list[i].shape[2])
            sub_data_dict['img'] = sub_img_list[i]
            sub_datas.append(sub_data_dict)
        return sub_datas, left_top_list


def read_json(json_path):
    with open(json_path) as json_file:
        data_json = json.load(json_file)
    return data_json

def get_camera_time(time, attr_json_path, camera_name):
    if os.path.getsize(attr_json_path) == 0:
        return None
    attr_dict = read_json(attr_json_path)
    sync_dict = attr_dict['sync']
    for idx, lidar_top_time in enumerate(sync_dict['lidar_top']):
        if lidar_top_time == int(time):
            return str(sync_dict[camera_name][idx])
    return None

def trans_sub_img_2_img(point, self_point, yaw):
    centor_point_x = self_point[0] + 200 * math.cos(yaw)
    centor_point_y = self_point[1] + 200 * math.sin(yaw)
    point[0] -= 600
    point[1] -= 600
    point[0] += centor_point_x
    point[1] += centor_point_y
    point_rotated_x = (point[0]-centor_point_x)*math.cos(yaw) - (point[1]-centor_point_y)*math.sin(yaw) + centor_point_x
    point_rotated_y = (point[0]-centor_point_x)*math.sin(yaw) + (point[1]-centor_point_y)*math.cos(yaw) + centor_point_y
    return [point_rotated_x, point_rotated_y]

def gen_video_for_checking(img_path, img_name, outdir_path):
    site_path = img_path.replace('/Bev_Result/Bev_img', '')
    site_name = site_path.split('/')[-1]
    bev_img_path = opj(img_path, img_name)
    out_img_list = []
    if os.path.exists(bev_img_path):
        bev_img = cv2.imread(bev_img_path)
        bev_json_path = opj(site_path, 'Bev_Result/Bev_json', site_name+'_bev.json')
        json_data = read_json(bev_json_path)
        centor_pt_x = json_data['center_pt'][0]
        centor_pt_y = json_data['center_pt'][1]
        halfmaplength = (json_data['imgH'] * json_data['solutionx'])/2
        solutionx = json_data['solutionx']
        solutiony = json_data['solutiony']
        clip_name_list = [clip_name for clip_name in json_data.keys() if 'clip' in clip_name]
        bev_img = bev_img[:,:3000,:]
        if len(clip_name_list)==0: return
        for idx, clip_name in enumerate(clip_name_list):
            point_color = 255-int(255 / (idx+1))
            packname = json_data[clip_name]['packname']
            if 'pose:' in json_data[clip_name].keys():
                pose_list = json_data[clip_name]['pose:']
            else:
                pose_list = json_data[clip_name]['pose']
            for pose_idx, pose_data in enumerate(pose_list):
                if pose_idx % 2 != 0: continue
                print('{}/{} {}/{}'.format(idx, len(clip_name_list), pose_idx, len(pose_list)))
                time = int(1000*pose_data[-1])
                self_pt_x = (pose_data[0] - centor_pt_x + halfmaplength) / solutionx
                self_pt_y = (pose_data[1] - centor_pt_y + halfmaplength) / solutiony
                self_pt_x = self_pt_x * (3000 / json_data['imgW'])
                self_pt_y = self_pt_y * (3000 / json_data['imgH'])
                if self_pt_x < 0 or self_pt_x > 2999 or self_pt_y < 0 or self_pt_y > 2999: continue

                vertex_point_0_0 = trans_sub_img_2_img([0, 0], [self_pt_x, self_pt_y], pose_data[6])
                vertex_point_1199_0 = trans_sub_img_2_img([1199, 0], [self_pt_x, self_pt_y], pose_data[6])
                vertex_point_0_1199 = trans_sub_img_2_img([0, 1199], [self_pt_x, self_pt_y], pose_data[6])
                vertex_point_1199_1199 = trans_sub_img_2_img([1199, 1199], [self_pt_x, self_pt_y], pose_data[6])

                vertex_point_0_0 = [round(x) for x in vertex_point_0_0]
                vertex_point_1199_0 = [round(x) for x in vertex_point_1199_0]
                vertex_point_0_1199 = [round(x) for x in vertex_point_0_1199]
                vertex_point_1199_1199 = [round(x) for x in vertex_point_1199_1199]

                flag_out_range = False
                for tmp in vertex_point_0_0 + vertex_point_1199_0 + vertex_point_0_1199 + vertex_point_1199_1199:
                    if tmp < 0 or tmp > 2999:
                        flag_out_range = True
                # if flag_out_range:
                #     continue

                bev_img = cv2.circle(bev_img, (round(self_pt_y), round(self_pt_x)), radius=10, color=(0, point_color, 255), thickness=6)

                big_img_drawed = cv2.line(copy.deepcopy(bev_img), tuple(vertex_point_0_0[::-1]), tuple(vertex_point_0_1199[::-1]), (0, 255, 0), thickness=6)
                big_img_drawed = cv2.line(big_img_drawed, tuple(vertex_point_0_1199[::-1]), tuple(vertex_point_1199_1199[::-1]), (0, 255, 0), thickness=6)
                big_img_drawed = cv2.line(big_img_drawed, tuple(vertex_point_1199_1199[::-1]), tuple(vertex_point_1199_0[::-1]), (0, 255, 0), thickness=6)
                big_img_drawed = cv2.line(big_img_drawed, tuple(vertex_point_1199_0[::-1]), tuple(vertex_point_0_0[::-1]), (0, 255, 0), thickness=6)

                ori_data_dir = opj(site_path, packname)
                attr_json_path = opj(ori_data_dir, 'attribute.json')
                camera_front_data_dir = opj(ori_data_dir, 'camera_front')
                camera_front_time = get_camera_time(time, attr_json_path, 'camera_front')
                camera_front = np.zeros((360, 640, 3))
                if camera_front_time is not None:
                    camera_front_path = opj(camera_front_data_dir, camera_front_time+'.jpg')
                    camera_front = cv2.imread(camera_front_path)
                camera_front = cv2.resize(camera_front, (640, 360))

                camera_front_left_data_dir = opj(ori_data_dir, 'camera_front_left')
                camera_front_left_time = get_camera_time(time, attr_json_path, 'camera_front_left')
                camera_front_left = np.zeros((360, 640, 3))
                if camera_front_left_time is not None:
                    camera_front_left_path = opj(camera_front_left_data_dir, camera_front_left_time+'.jpg')
                    camera_front_left = cv2.imread(camera_front_left_path)
                camera_front_left = cv2.resize(camera_front_left, (640, 360))

                camera_front_right_data_dir = opj(ori_data_dir, 'camera_front_right')
                camera_front_right_time = get_camera_time(time, attr_json_path, 'camera_front_right')
                camera_front_right = np.zeros((360, 640, 3))
                if camera_front_right_time is not None:
                    camera_front_right_path = opj(camera_front_right_data_dir, camera_front_right_time+'.jpg')
                    camera_front_right = cv2.imread(camera_front_right_path)
                camera_front_right = cv2.resize(camera_front_right, (640, 360))

                camera_rear_data_dir = opj(ori_data_dir, 'camera_rear')
                camera_rear_time = get_camera_time(time, attr_json_path, 'camera_rear')
                camera_rear = np.zeros((360, 640, 3))
                if camera_rear_time is not None:
                    camera_rear_path = opj(camera_rear_data_dir, camera_rear_time+'.jpg')
                    camera_rear = cv2.imread(camera_rear_path)
                camera_rear = cv2.resize(camera_rear, (640, 360))

                camera_rear_left_data_dir = opj(ori_data_dir, 'camera_rear_left')
                camera_rear_left_time = get_camera_time(time, attr_json_path, 'camera_rear_left')
                camera_rear_left = np.zeros((360, 640, 3))
                if camera_rear_left_time is not None:
                    camera_rear_left_path = opj(camera_rear_left_data_dir, camera_rear_left_time+'.jpg')
                    camera_rear_left = cv2.imread(camera_rear_left_path)
                camera_rear_left = cv2.resize(camera_rear_left, (640, 360))

                camera_rear_right_data_dir = opj(ori_data_dir, 'camera_rear_right')
                camera_rear_right_time = get_camera_time(time, attr_json_path, 'camera_rear_right')
                camera_rear_right = np.zeros((360, 640, 3))
                if camera_rear_right_time is not None:
                    camera_rear_right_path = opj(camera_rear_right_data_dir, camera_rear_right_time+'.jpg')
                    camera_rear_right = cv2.imread(camera_rear_right_path)
                camera_rear_right = cv2.resize(camera_rear_right, (640, 360))

                # bev_img_resized = cv2.resize(big_img_drawed, (1920, 1920))
                bev_img_resized = cv2.resize(big_img_drawed, (960, 960))
                bev_img_zero_placeholder = np.zeros((960,480,3))
                bev_img_resized = np.concatenate((bev_img_zero_placeholder, bev_img_resized, bev_img_zero_placeholder), axis=1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                up_imgs = np.concatenate((camera_front_left, camera_front, camera_front_right), axis=1)
                down_imgs = np.concatenate((camera_rear_left, camera_rear, camera_rear_right), axis=1)
                out_img = np.concatenate((up_imgs, down_imgs, bev_img_resized), axis=0)
                out_img = cv2.putText(out_img, '{}/'.format(site_path), (50,50), font, 1, (255,255,255), 2)
                out_img = cv2.putText(out_img, '{}/{}'.format(packname, time), (50,120), font, 1, (255,255,255), 2)
                h = out_img.shape[0]
                w = out_img.shape[1]
                out_img_list.append(out_img.astype(np.uint8))
        out_path = opj(outdir_path, '{}.avi'.format(site_name))
        fps = 25
        try:
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        except:
            return
        for i in range(len(out_img_list)):
            print('gen video {}/{}'.format(i, len(out_img_list)))
            out.write(out_img_list[i])
        out.release()
        dir = out_path.strip(".avi")
        init_command = 'hitc init -t eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNTc1NjgwODcsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InNoaXFpLnh1IiwiT3JnYW5pemF0aW9uIjoicmVndWxhci1lbmdpbmVlciIsIk9yZ2FuaXphdGlvbklEIjoxfQ.HOrzZ8Hu_9Q0PAisdtqtL9hhy1lTZjZHpqCpXqGHf3D3VF6kuyqUUWG1Q46ZSgqer2E7ZN2lqgJ2W9mS8AhTvVfRDYw_kMYGJEK4f8Wd1PUPBXBYfdM52N1iBUEdoyvTiUwah91shWFfPCu1tQMVz9MHwk9d_rGB4rYRw4rrsdUTq-WOyZQ_PqtUgCFzuq9wvxO8QLbJtcTGpZtChujtxchBWV-5o2H0xRqG5nNMM8SvPtjVoTq70fcX4QkUiweAtcejWTQiad2KNgcfi_H6-_ipF6FKRIXI5kqe3KuvUQSgOpBTMkIVw8bMAT2_bZKzOHfeO203nvGmQGbIogmJeQ'
        os.system(init_command)
        bucket_out_path = out_path.replace('/bucket/output/', '').strip(".avi")
        print('hitc bkt-file rm dmpv2://{}.avi -y'.format(bucket_out_path))
        command = "ffmpeg -i %s.avi %s.mp4 -y && hitc bkt-file rm dmpv2://%s.avi -y" % (dir, dir, bucket_out_path)
        os.system(command)
    return
