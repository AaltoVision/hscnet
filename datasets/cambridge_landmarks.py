from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data
import scipy.io as spio

from .utils import *


class Cambridge(data.Dataset):
    def __init__(self, root, dataset='Cambridge', scene='GreatCourt', 
                split='train', model='hscnet', aug='True'):

        self.intrinsics_color = np.array([[744.375, 0.0,     426.0],
                       [0.0,     744.375, 240.0],
                       [0.0,     0.0,  1.0]])
                       
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)    
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.scene = scene
        self.root = os.path.join(root,'Cambridge', self.scene)
        self.split = split
        self.obj_suffixes = ['.color.png', '.pose.txt', '.depth.png', 
                    '.label.png']
        self.obj_keys = ['color', 'pose', 'depth', 'label']
                    
        self.centers = np.load(os.path.join(self.root, 'centers.npy'))

        with open(os.path.join(self.root, '{}{}'.format(self.split, '.txt')), 
                'r') as f:
            self.frames = f.readlines()
            
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')

        obj_files = ['{}{}'.format(frame, 
                obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, self.split, 
                obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
             
        img = cv2.imread(objs['color']) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (852, 480)) 
        pose = np.loadtxt(objs['pose'])

        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose
        
        lbl = cv2.imread(objs['label'],-1) 
        ctr_coord = self.centers[np.reshape(lbl,(-1))-1,:]
        ctr_coord = np.reshape(ctr_coord,(480,852,3)) * 1000
        
        depth_fn = objs['depth']
        if self.scene != 'ShopFacade':
            depth_fn = depth_fn.replace('.png','.tiff')
    
        depth = cv2.imread(depth_fn,-1)      
            
        pose[0:3,3] = pose[0:3,3] * 1000
            
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)

        img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord,
                mask, lbl, self.aug)

        if self.model == 'hscnet':
            coord = coord - ctr_coord
        
        img_h, img_w = img.shape[0:2]
        th, tw = 480, 640
        x1 = random.randint(0, img_w - tw)
        y1 = random.randint(0, img_h - th)
        
        img = img[y1:y1+th,x1:x1+tw,:]
        coord = coord[y1:y1+th,x1:x1+tw,:]
        mask = mask[y1:y1+th,x1:x1+tw]
        lbl = lbl[y1:y1+th,x1:x1+tw]
        
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.float16)

        lbl_1 = (lbl - 1) // 25
        lbl_2 = ((lbl - 1) % 25) 
        
        img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img, 
                coord, mask, lbl_1, lbl_2)

        return img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh  
