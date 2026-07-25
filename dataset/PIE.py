import os
from pathlib import Path

import json
import pickle as pkl
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from SC_MSF.structures.trajectory_ops import *
from SC_MSF.utils.box_utils import signedIOU
from dataset.PIE_origin import PIE
from . import transforms as T
from torchvision.transforms import functional as F
import copy
from SC_MSF.utils.dataset_utils import bbox_to_goal_map, squarify, img_pad
import glob
import time
import pdb
from .selection import select_key_flow_sequence, select_key_poses

class PIEDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.root = cfg.DATASET.ROOT
        self.cfg = cfg
        # NOTE: add downsample function
        self.downsample_step = int(30/self.cfg.DATASET.FPS)
        traj_data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

        traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': [],
                       'prediction_type': ['bbox']
                       }
        imdb = PIE(data_path=self.root)

        traj_model_opts['enc_input_type'].extend(['obd_speed', 'heading_angle'])
        traj_model_opts['prediction_type'].extend(['obd_speed', 'heading_angle'])
        beh_seq = imdb.generate_data_trajectory_sequence(self.split, **traj_data_opts)
        self.data = self.get_traj_data(beh_seq, **traj_model_opts)

    def __getitem__(self, index):
        obs_bbox = torch.FloatTensor(self.data['obs_bbox'][index])
        heading_angle = torch.FloatTensor(self.data['heading_angle'][index])
        obd_speed = torch.FloatTensor(self.data['obd_speed'][index])
        pred_bbox = torch.FloatTensor(self.data['pred_bbox'][index])
        cur_image_file = self.data['obs_image'][index][-1]
        image_file = self.data['obs_image'][index]
        # print(cur_image_file,image_file)
#####################################################################
        obs_pose, key_pose = select_key_poses(self.data['obs_pose'][index])
        ########################################################################
        obs_flow = np.array(self.data['obs_flow'][index])
        key_flow_seq = select_key_flow_sequence(obs_flow)

        # print(mean_flow)
        #########################################################3
            ####################################################### 处理图像信息
        obs_img = np.array(self.data['obs_img'][index])
        #######################################################
        obs_img = torch.FloatTensor(obs_img)
        obs_flow = torch.FloatTensor(obs_flow)
        key_flow_seq = torch.FloatTensor(key_flow_seq)
        obs_pose = torch.FloatTensor(obs_pose)
        cur_pose = torch.FloatTensor(obs_pose)[-1]
        str_pose = torch.FloatTensor(obs_pose)[0]
        key_pose = torch.FloatTensor(key_pose)
        ########################################################s输入评价权重
        pred_resolution = torch.FloatTensor(self.data['pred_resolution'][index])
        obs_pid = self.data['obs_pid'][index]
        # flow_input = torch.FloatTensor(self.data['flow_input'][index])
        ########################################################确定车速和角度极值
        # ret = {'input_x':obs_bbox, 'flow_input':flow_input,
        #        'target_y':pred_bbox, 'cur_image_file':cur_image_file}
        ret = {'input_x':obs_bbox,
               'pose' : obs_pose, # 最终返回的
               'image_features': obs_img,
               'flow': obs_flow,
               'key_flow_seq' : key_flow_seq,
               'cur_pose' :cur_pose,
               'str_pose': str_pose,
               'key_pose' : key_pose,
                'target_y':pred_bbox,
               'cur_image_file':cur_image_file,
               'image_file':image_file,
                'obs_pid': obs_pid,
               'obd_speed': obd_speed,
               'heading_angle': heading_angle}

        ret['timestep'] = int(Path(cur_image_file).stem)

        ret['pred_resolution'] = pred_resolution
        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def get_traj_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):
        """
        Generates tracks by sampling from pedestrian sequences
        :param dataset: The raw data passed to the method
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on dataset. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
        :param observe_length: The length of the observation (i.e. time steps of the encoder)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
        :return: A dictinary containing sampled tracks for each data modality
        """
        #  Calculates the overlap in terms of number of frames
        seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except:# KeyError:
                raise KeyError('Wrong data type is selected %s' % dt)

        d['image'] = dataset['image']
        d['pid'] = dataset['pid']
        d['resolution'] = dataset['resolution']
        ##############################################################################################
        d['flow'] = dataset['flow']
        d['cnn'] = dataset['cnn']
        d['pose'] = dataset['pose']

        num_trks = len(d['image'])
        traj_root = self.cfg.DATASET.TRAJECTORY_PATH
        # for images, pids in zip(d['image'], d['pid']):
        #     sid = images[0].split('/')[-3]
        #     vid = images[0].split('/')[-2]
        #     pid = pids[0][0]
        #     traj_name = sid + '_' + vid + '_' + pid
        #     traj_path = os.path.join(traj_root, traj_name+'.pkl')
        #     traj = pkl.load(open(traj_path, 'rb'))
        #     if len(traj['flow']) != len(images):
        #         pdb.set_trace()
        #     d['flow'].append(traj['flow'])

        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            for track in d[k]:
                for i in range(0, len(track) - seq_length + 1, overlap_stride):
                    tracks.append(track[i:i + seq_length])
            d[k] = tracks
        #  Normalize tracks using FOL paper method,
        d['bbox'] = self.convert_normalize_bboxes(d['bbox'], d['resolution'],
                                                  self.cfg.DATASET.NORMALIZE, self.cfg.DATASET.BBOX_TYPE)
        return d

    def convert_normalize_bboxes(self, all_bboxes, all_resolutions, normalize, bbox_type):
        '''input box type is x1y1x2y2 in original resolution'''
        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = np.array(all_bboxes[i])
            # NOTE ltrb to cxcywh
            if bbox_type == 'cxcywh':
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]]/2
            # NOTE Normalize bbox
            if normalize == 'zero-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (bbox - _min) / (_max - _min)
            elif normalize == 'plus-minus-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            elif normalize == 'none':
                pass
            else:
                raise ValueError(normalize)
            all_bboxes[i] = bbox
        return all_bboxes

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        :param data: A dictionary of different data types
        :param data_type: The data types defined for encoder and decoder input/output
        :return: A unified data representation as a list
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))

        #  Concatenate different data points into a single representation
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        elif len(d) == 1:
            return d[0]
        else:
            return d

    def get_traj_data(self, data, **model_opts):
        """
        Main data generation function for training/testing
        :param data: The raw data
        :param model_opts: Control parameters for data generation characteristics (see below for default values)
        :return: A dictionary containing training and testing data
        """

        opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,
            'observe_length': self.cfg.MODEL.INPUT_LEN,
            'predict_length': self.cfg.MODEL.PRED_LEN,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }
        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_traj_tracks(data, data_types, observe_length,
                                      opts['predict_length'], opts['track_overlap'],
                                      opts['normalize_bbox'])
        obs_slices = {}
        pred_slices = {}
        #  Generate observation/prediction sequences from the tracks
        for k in data_tracks.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            # NOTE: Add downsample function
            down = self.downsample_step
            obs_slices[k].extend([d[down-1:observe_length:down] for d in data_tracks[k]])
            pred_slices[k].extend([d[observe_length+down-1::down] for d in data_tracks[k]])

        ret =  {'obs_image': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'obs_resolution': obs_slices['resolution'],
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'pred_resolution': pred_slices['resolution'],
                'obs_bbox': np.array(obs_slices['bbox']), #enc_input,

                'pred_bbox': np.array(pred_slices['bbox']), #pred_target,
                'obs_pose':obs_slices['pose'],
                'obs_flow': obs_slices['flow'],
                'obs_img': obs_slices['cnn'],
                'obd_speed' :obs_slices['obd_speed'],
                'heading_angle':obs_slices['heading_angle']
                }

        return ret

    def get_path(self,
                 file_name='',
                 save_folder='models',
                 dataset='pie',
                 model_type='trajectory',
                 save_root_folder='data/'):
        """
        A path generator method for saving model and config data. It create directories if needed.
        :param file_name: The actual save file name , e.g. 'model.h5'
        :param save_folder: The name of folder containing the saved files
        :param dataset: The name of the dataset used
        :param save_root_folder: The root folder
        :return: The full path for the model name and the path to the final folder
        """
        save_path = os.path.join(save_root_folder, dataset, model_type, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path
