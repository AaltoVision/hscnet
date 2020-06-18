from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data
import scipy.io as spio

from .utils import *


class TwelveScenes(data.Dataset):
    def __init__(self, root, dataset='12S', scene='apt2/bed', split='train', 
                    model='hscnet', aug='True'):
        self.intrinsics_color = np.array([[572.0, 0.0,     320.0],
                       [0.0,     572.0, 240.0],
                       [0.0,     0.0,  1.0]])
                       
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.root = os.path.join(root,'12Scenes')
        self.scene = scene
        if self.dataset == '12S':
            self.centers = np.load(os.path.join(self.root, scene,
                            'centers.npy'))
        else: 
            self.scenes = ['apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']
            self.transl = [[0,-20,0],[0,-20,0],[20,0,0],[20,0,0],[25,0,0],
                    [20,0,0],[-20,0,0],[-25,5,0],[-20,0,0],[-20,-5,0],[0,20,0],
                    [0,20,0]]
            if self.dataset == 'i12S':
                self.ids = [0,1,2,3,4,5,6,7,8,9,10,11]
            else:
                self.ids = [7,8,9,10,11,12,13,14,15,16,17,18]
            self.scene_data = {}
            for scene, t, d in zip(self.scenes, self.transl, self.ids):
                self.scene_data[scene] = (t, d, np.load(os.path.join(self.root,
                    scene,  'centers.npy')))

        self.split = split
        self.obj_suffixes = ['.color.jpg', '.pose.txt', '.depth.png', 
                '.label.png']
        self.obj_keys = ['color', 'pose', 'depth', 'label']
        
        if self.dataset == '12S' or self.split == 'test':
            with open(os.path.join(self.root, self.scene, 
                    '{}{}'.format(self.split, '.txt')), 'r') as f:
                self.frames = f.readlines()
        else:
            self.frames = []
            for scene in self.scenes:
                with open(os.path.join(self.root, scene, 
                        '{}{}'.format(self.split, '.txt')), 'r') as f:
                    frames = f.readlines()
                    frames = [scene + ' ' + frame for frame in frames ]
                self.frames.extend(frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')

        if self.dataset != '12S' and self.split == 'train':
            scene, frame = frame.split(' ')
            centers = self.scene_data[scene][2] 
        else: 
            scene = self.scene
            if self.split == 'train':
                centers = self.centers
        
        obj_files = ['{}{}'.format(frame, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, scene, 'data', 
                    obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
        
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))

        pose = np.loadtxt(objs['pose'])
        if self.dataset != '12S' and (self.model != 'hscnet' \
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
    
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)

        img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord, 
                mask, lbl, self.aug)
        
        if self.model == 'hscnet':
            coord = coord - ctr_coord
               
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.float16)
       
        if  self.dataset=='12S':
            lbl_1 = (lbl - 1) // 25
        else:
            lbl_1 = (lbl - 1) // 25 + 25*self.scene_data[scene][1]
        lbl_2 = ((lbl - 1) % 25) 
        
        if  self.dataset=='12S':
            N1=25
        if  self.dataset=='i12S':
            N1=300
        if  self.dataset=='i19S':
            N1=475
        
        img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img, 
                    coord, mask, lbl_1, lbl_2, N1)

        return img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh        
