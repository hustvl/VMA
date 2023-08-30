import numpy as np
from os import path as osp
from PIL import Image
import torch
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import mmcv
from shapely.geometry import LineString
from .mean_ap import average_precision
from .tpfp import tpfp_gen, custom_tpfp_gen, custom_tpfp_gen_1
from sklearn.metrics import precision_score,f1_score,recall_score
from mmcv.utils import print_log
from terminaltables import AsciiTable
import json
import os
from scipy.spatial import cKDTree
from skimage import measure
from scipy.sparse.csgraph import dijkstra
import pickle
import multiprocessing
from functools import partial

def update_graph(start_vertex,end_vertex,graph,set_value=1):
    start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])])
    end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
    p = start_vertex
    d = end_vertex - start_vertex
    N = np.max(np.abs(d))
    graph[start_vertex[0],start_vertex[1]] = set_value
    graph[end_vertex[0],end_vertex[1]] = set_value
    if N:
        s = d / (N)
        for i in range(0,N):
            p = p + s
            graph[int(round(p[0])),int(round(p[1]))] = set_value
    return graph

def eval_metric(graph, mask, name, gt_data):
    def tuple2list(t):
        return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

    graph = graph.cpu().detach().numpy().squeeze()
    gt_image = mask
    gt_points = tuple2list(np.where(gt_image!=0))
    graph_points = tuple2list(np.where(graph!=0))
    if not len(graph_points):
        return [0, 0, 0], [0, 0, 0], [0, 0, 0], 0, 0, 0, 1
    graph_acc = []
    graph_recall = []
    r_f = []
    apls = 0
    entropy_conn = 0
    naive_conn = 0
    gt_tree = cKDTree(gt_points)
    for c_i, thre in enumerate([2, 5, 10]):
        graph_acc_num = 0
        graph_recall_num = 0
        if len(graph_points):
            graph_tree = cKDTree(graph_points)
            dis_gt2graph,_ = graph_tree.query(gt_points, k=1)
            dis_graph2gt,_ = gt_tree.query(graph_points, k=1)
            graph_recall_num = len([x for x in dis_gt2graph if x<thre])/len(dis_gt2graph)
            graph_acc_num = len([x for x in dis_graph2gt if x<thre])/len(dis_graph2gt)
        if graph_acc_num*graph_recall_num:
            r_f_num = 2*graph_recall_num * graph_acc_num / (graph_acc_num+graph_recall_num)
        else:
            r_f_num = 0
        graph_acc.append(graph_acc_num)
        graph_recall.append(graph_recall_num)
        r_f.append(r_f_num)
    
    pre_image = graph
    gt_instance_map = measure.label(gt_image / 255,background=0)
    gt_instance_indexes = np.unique(gt_instance_map)[1:]
    gt_assigned_lengths = [[] for x in range(len(gt_instance_indexes))]
    gt_instance_length = []
    gt_instance_points = []
    gt_covered = []
    for index in gt_instance_indexes:
        instance_map = (gt_instance_map == index)
        instance_points = np.where(instance_map==1)
        instance_points = [[instance_points[0][i],instance_points[1][i]] for i in range(len(instance_points[0]))]
        gt_instance_length.append(len(instance_points))
        gt_covered.append(np.zeros((len(instance_points))))
        gt_instance_points.append(instance_points)
    pre_instance_map = measure.label(pre_image, background=0)
    pre_instance_indexes = np.unique(pre_instance_map)[1:]
    gt_points = np.where(gt_image!=0)
    gt_points = [[gt_points[0][i],gt_points[1][i]] for i in range(len(gt_points[0]))]
    tree = cKDTree(gt_points)
    # each pre_index is an predicted instance
    for index in pre_instance_indexes: 
        votes = []
        instance_map = (pre_instance_map == index)
        instance_points = np.where(instance_map==1)
        instance_points = [[instance_points[0][i],instance_points[1][i]] for i in range(len(instance_points[0]))]
        if instance_points:
            # Each predicted point of the current pre-instance finds its closest gt point and votes
            # to the gt-instance that the closest gt point belongs to.
            _, iis = tree.query(instance_points,k=[1])
            closest_gt_points = [[gt_points[x[0]][0],gt_points[x[0]][1]] for x in iis]
            votes = [gt_instance_map[x[0],x[1]] for x in closest_gt_points]
        # count the voting results
        votes_summary = np.zeros((len(gt_instance_indexes)))
        for j in range(len(gt_instance_indexes)):
            # the number of votes made to gt-instance j+1
            votes_summary[j] = votes.count(j+1) 
        # find the gt-instance winning the most vote and assign the current pre-instance to it
        if np.max(votes_summary):
            vote_result = np.where(votes_summary==np.max(votes_summary))[0][0]
            # the length of the pre-instance assigned to corresponding gt-instance 
            gt_assigned_lengths[vote_result].append(len(instance_points))
            # calculate projection of the predicted instance to corresponding gt-instance
            instance_tree = cKDTree(gt_instance_points[vote_result])
            _, iis = instance_tree.query(instance_points,k=[1])
            gt_covered[vote_result][np.min(iis):np.max(iis)+1] = 1
    
    # calculate ECM
    # iterate all gt-instances, calculate connectivity of each of them 
    for j,lengths in enumerate(gt_assigned_lengths):
        # lengths are the length of assigned pre-instances to the current gt-instance
        if len(lengths):
            lengths = np.array(lengths)
            # contribution of each assigned pre-instance
            probs = (lengths / np.sum(lengths)).tolist()
            C_j = 0
            for p in probs:
                C_j += -p*np.log2(p)
            entropy_conn += np.exp(-C_j) * np.sum(gt_covered[j]) / len(gt_points)
            naive_conn += 1 / len(lengths)
    if len(gt_assigned_lengths):
        naive_conn = naive_conn / len(gt_assigned_lengths)
    
    apls_counter = 0
    pre_data = generate_graph(graph)
    adjacent = pre_data['adj']
    pre_points = pre_data['vertices']
    if len(pre_points):
        
        tree = cKDTree(pre_points)
        for gt_data_instance in gt_data:
            gt_points_instance = gt_data_instance['vertices']
            gt_adjacent = gt_data_instance['adj']
            if len(gt_points_instance) > 5:
                # randomly find some pairs of points for APLS calculation
                for ii in range(len(gt_points_instance)//3):
                    select_index = np.random.choice(len(gt_points_instance),2,replace=False)
                    init_v = gt_points_instance[select_index[0]]
                    end_v = gt_points_instance[select_index[1]]
                    source = select_index[0]
                    target = select_index[1]
                    dist_matrix = dijkstra(gt_adjacent, directed=False, indices=[source],
                                            unweighted=False)
                    gt_length = dist_matrix[0,target]
                    # find corresponding pre_vertex by min-Euclidean distance
                    dds,iis = tree.query([init_v,end_v],k=1)
                    source = iis[0]
                    target = iis[1]
                    # vertex too far away is treated as not reachable
                    if np.max(dds) < 50:
                        dist_matrix = dijkstra(adjacent, directed=False, indices=[source],
                                            unweighted=False)
                        pre_length = dist_matrix[0,target]
                    else:
                        pre_length = 10000
                    apls += min(1,abs(gt_length - pre_length) / (gt_length))
                    apls_counter += 1
    
    if apls_counter:
        apls = 1 - apls / apls_counter
    return graph_acc, graph_recall, r_f, entropy_conn, naive_conn, apls, 0

class Vertex():
    def __init__(self,v):
        self.coord = v
        self.index = v[0] * 1000 + v[1]
        self.neighbors = []
        self.unprocessed_neighbors = []
        self.processed_neighbors = []
        self.sampled_neighbors = []
        self.key_vertex = False
    def compare(self,v):
        if self.coord[0] == v[0] and self.coord[1] == v[1]:
            return True
        return False
    def next(self,previous):
        neighbors = self.neighbors.copy()
        neighbors.remove(previous)
        return neighbors[0]
    def distance(self,v):
        return pow(pow(self.coord[0]-v.coord[0],2)+pow(self.coord[1]-v.coord[1],2),0.5)

class Graph():
    def __init__(self):
        self.vertices = []
        self.key_vertices = []
        self.sampled_vertices = []
    def find_vertex(self,index):
        for v in self.vertices:
            if index == v.index:
                return v
        return None
    def add_v(self,v,neighbors):
        self.vertices.append(v)
        for n in neighbors:
            index = n[0] * 1000 + n[1]
            u = self.find_vertex(index)
            if u is not None:
                u.neighbors.append(v)
                v.neighbors.append(u)
                u.unprocessed_neighbors.append(v)
                v.unprocessed_neighbors.append(u)
    def find_key_vertices(self):
        for v in self.vertices:
            if len(v.neighbors)!=2:
                v.key_vertex = True
                self.key_vertices.append(v)
                self.sampled_vertices.append(v)

def generate_graph(skeleton):
    def find_neighbors(v,img,remove=False):
        output_v = []
        def get_pixel_value(u):
            if max(u) > 999 or min(u) < 0:
                return
            if img[u[0],u[1]]:
                output_v.append(u)

        get_pixel_value([v[0]+1,v[1]])
        get_pixel_value([v[0]-1,v[1]])
        get_pixel_value([v[0],v[1]-1])
        get_pixel_value([v[0],v[1]+1])
        get_pixel_value([v[0]+1,v[1]-1])
        get_pixel_value([v[0]+1,v[1]+1])
        get_pixel_value([v[0]-1,v[1]-1])
        get_pixel_value([v[0]-1,v[1]+1])
        if remove:
            img[v[0],v[1]] = 0
        return output_v

    graph = Graph()
    img = skeleton
    pre_points = np.where(img!=0)
    pre_points = [[pre_points[0][i],pre_points[1][i]] for i in range(len(pre_points[0]))]
    for point in pre_points:
        v = Vertex(point)
        graph.add_v(v,find_neighbors(point,img))
    graph.find_key_vertices()
    for key_vertex in graph.key_vertices:
        if len(key_vertex.unprocessed_neighbors):
            for neighbor in key_vertex.unprocessed_neighbors:
                key_vertex.unprocessed_neighbors.remove(neighbor)
                #
                curr_v = neighbor
                pre_v = key_vertex
                sampled_v = key_vertex
                counter = 1
                while(not curr_v.key_vertex):
                    if counter % 30 == 0:
                        sampled_v.sampled_neighbors.append(curr_v)
                        curr_v.sampled_neighbors.append(sampled_v)
                        sampled_v = curr_v
                        if not sampled_v.key_vertex:
                            graph.sampled_vertices.append(sampled_v)
                    next_v = curr_v.next(pre_v)
                    pre_v = curr_v
                    curr_v = next_v
                    counter += 1
                sampled_v.sampled_neighbors.append(curr_v)
                curr_v.sampled_neighbors.append(sampled_v)
                curr_v.unprocessed_neighbors.remove(pre_v)
    
    adjacent = np.ones((len(graph.sampled_vertices),len(graph.sampled_vertices))) * np.inf
    vertices = []
    for ii, v in enumerate(graph.sampled_vertices):
        v.index = ii
        vertices.append([int(v.coord[0]),int(v.coord[1])])
    for v in graph.sampled_vertices:
        for u in v.sampled_neighbors:
            dist = v.distance(u)
            adjacent[v.index,u.index] = dist
            adjacent[u.index,v.index] = dist
    return {'vertices':vertices,'adj':adjacent}

def evaluate_single(pred_result, mask_dir):
    
    pred_instances = pred_result['pred_instances']
    image_name = pred_result['image_path'].split('/')[-1]
    if image_name.endswith('tiff'):
        image_name = image_name.replace('tiff', 'png')
    else:
        image_name = image_name.replace('jpg', 'png')
    mask_image_path = osp.join(mask_dir, image_name)
    if image_name.endswith('png'):
        sample_seq_path = mask_image_path.replace('binary_map', 'sampled_seq').replace('png', 'pickle')
    else:
        sample_seq_path = mask_image_path.replace('binary_map', 'sampled_seq').replace('jpg', 'pickle')
    
    with open(sample_seq_path,'rb') as jf:
        gt_data = pickle.load(jf)
    mask = np.array(Image.open(mask_image_path))[:,:,0]
    graph = torch.zeros(1000,1000)
    for instance in pred_instances:
        for i in range(1, len(instance['data'])):
            # import pdb;pdb.set_trace()
            start_vertex, end_vertex = instance['data'][i-1], instance['data'][i]
            graph = update_graph(start_vertex, end_vertex, graph)
    return eval_metric(graph, mask, image_name, gt_data)

def evaluate_all(results_file, mask_dir):
    with open(results_file, 'r') as f:
        pred_results = json.load(f)
    total_image = len(pred_results)
    graph_acc, graph_recall, r_f =[0, 0, 0], [0, 0, 0], [0, 0, 0]
    ECM, Naive, APLS = 0, 0, 0
    results = []
    for pred_result in tqdm(pred_results.values(), desc='evaluate metric'):
        results.append(evaluate_single(pred_result, mask_dir))
    # with multiprocessing.Pool(processes = 16) as pool:
    #     partial_func = partial(evaluate_single, mask_dir=mask_dir)
    #     results = list(tqdm(pool.imap(partial_func, pred_results.values()), total=len(pred_results.values()), desc='evaluate the metric'))
    for acc_single, recall_single, r_f_single, entropy_conn, naive_conn, apls, non_pred_image in results:
        graph_acc  = [i + j for i, j in zip(graph_acc, acc_single)]
        graph_recall = [i + j for i, j in zip(graph_recall, recall_single)]
        r_f = [i + j for i, j in zip(r_f, r_f_single)]
        ECM += entropy_conn
        Naive += naive_conn
        APLS += apls
        total_image -= non_pred_image
    

    graph_acc_1, graph_recall_1, r_f_1 = graph_acc[0]/total_image, graph_recall[0]/total_image, r_f[0]/total_image
    graph_acc_2, graph_recall_2, r_f_2 = graph_acc[1]/total_image, graph_recall[1]/total_image, r_f[1]/total_image
    graph_acc_5, graph_recall_5, r_f_5 = graph_acc[2]/total_image, graph_recall[2]/total_image, r_f[2]/total_image
    ECM = ECM / total_image
    Naive = Naive /total_image
    APLS = APLS /total_image

    return {'accuracy_1':graph_acc_1, 'accuracy_2':graph_acc_2, 'accuracy_5':graph_acc_5, \
            'recall_1':graph_recall_1, 'recall_2':graph_recall_2, 'recall_5':graph_recall_5, \
            'f1_score_1': r_f_1, 'f1_score_2': r_f_2, 'f1_score_5': r_f_5, 'ECM':ECM, 'Naive':Naive, 'APLS':APLS}