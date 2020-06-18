from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data
import scipy.io as spio

from .utils import *


class SevenScenes(data.Dataset):
    def __init__(self, root, dataset='7S', scene='heads', split='train', 
                    model='hscnet', aug='True'):
        self.intrinsics_color = np.array([[525.0, 0.0,     320.0],
                       [0.0,     525.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_depth = np.array([[585.0, 0.0,     320.0],
                       [0.0,     585.0, 240.0],
                       [0.0,     0.0,  1.0]])
        
        self.intrinsics_depth_inv = np.linalg.inv(self.intrinsics_depth)
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.root = os.path.join(root,'7Scenes')
        self.calibration_extrinsics = np.loadtxt(os.path.join(self.root, 
                        'sensorTrans.txt'))
        self.scene = scene
        if self.dataset == '7S':
            self.scene_ctr = np.loadtxt(os.path.join(self.root, scene,
                            'translation.txt'))
            self.centers = np.load(os.path.join(self.root, scene,
                            'centers.npy'))
        else: 
            self.scenes = ['chess','fire','heads','office','pumpkin',
                            'redkitchen','stairs']
            self.transl = [[0,0,0],[10,0,0],[-10,0,0],[0,10,0],[0,-10,0],
                            [0,0,10],[0,0,-10]]
            self.ids = [0,1,2,3,4,5,6]
            self.scene_data = {}
            for scene, t, d in zip(self.scenes, self.transl, self.ids):
                self.scene_data[scene] = (t, d, np.load(os.path.join(self.root,
                    scene,  'centers.npy')),
                    np.loadtxt(os.path.join(self.root, 
                    scene,'translation.txt')))

        self.split = split
        self.obj_suffixes = ['.color.png','.pose.txt', '.depth.png',
                '.label.png']
        self.obj_keys = ['color','pose', 'depth','label']
                    
        with open(os.path.join(self.root, '{}{}'.format(self.split, 
                '.txt')), 'r') as f:
            self.frames = f.readlines()
            if self.dataset == '7S' or self.split == 'test':
                self.frames = [frame for frame in self.frames \
                if self.scene in frame]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')
        scene, seq_id, frame_id = frame.split(' ') 

        if self.dataset!='7S':
            centers = self.scene_data[scene][2] 
            scene_ctr = self.scene_data[scene][3] 
        else:
            centers = self.centers
            scene_ctr = self.scene_ctr   

        obj_files = ['{}{}'.format(frame_id, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, scene, 
                    seq_id, obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
       
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pose = np.loadtxt(objs['pose'])

        pose[0:3,3] = pose[0:3,3] - scene_ctr
        
        if self.dataset != '7S' and (self.model != 'hscnet' \
                        or self.split == 'test'):
            pose[0:3,3] = pose[0:3,3] + np.array(self.scene_data[scene][0]) 

        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose        
        
        lbl = cv2.imread(objs['label'],-1)
        
        ctr_coord = centers[np.reshape(lbl,(-1))-1,:]
        ctr_coord = np.reshape(ctr_coord,(480,640,3)) * 1000
        
        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
     
        depth[depth==65535] = 0
        depth = depth * 1.0
        depth = get_depth(depth, self.calibration_extrinsics, 
            self.intrinsics_color, self.intrinsics_depth_inv)
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)
        
        img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord, 
                mask, lbl, self.aug)
        
        if self.model == 'hscnet':
            coord = coord - ctr_coord
    
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.float16)
       
        if  self.dataset=='7S':
            lbl_1 = (lbl - 1) // 25
        else:
            lbl_1 = (lbl - 1)//25 + 25*self.scene_data[scene][1]
        lbl_2 = ((lbl - 1) % 25) 
        
        if  self.dataset=='7S':
            N1=25
        if  self.dataset=='i7S':
            N1=175
        if  self.dataset=='i19S':
            N1=475
        
        img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img, 
                coord, mask, lbl_1, lbl_2, N1)

        return img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh        
