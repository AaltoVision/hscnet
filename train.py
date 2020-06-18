from __future__ import division

import sys
import os
import random
import argparse
from pathlib import Path
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm

from models import get_model
from datasets import get_dataset
from loss import *
from utils import *

def train(args):
    
    # prepare datasets
    if args.dataset == 'i19S':
        datasetSs = get_dataset('7S')
        datasetTs = get_dataset('12S')
        datasetSs = datasetSs(args.data_path, args.dataset, model=args.model,
                    aug=args.aug)
        datasetTs = datasetTs(args.data_path, args.dataset, model=args.model,
                    aug=args.aug)
        dataset = data.ConcatDataset([datasetSs,datasetTs])
    else:
        if args.dataset in ['7S', 'i7S']: 
            dataset = get_dataset('7S')
        if args.dataset in ['12S', 'i12S']: 
            dataset = get_dataset('12S')
        if args.dataset == 'Cambridge': 
            dataset = get_dataset('Cambridge')
        dataset = dataset(args.data_path, args.dataset, args.scene,
                          model=args.model, aug=args.aug)

    trainloader = data.DataLoader(dataset, batch_size=args.batch_size,
                                  num_workers=4, shuffle=True)
    
    # loss
    reg_loss = EuclideanLoss()
    if args.model == 'hscnet':
        cls_loss = CELoss()
        if args.dataset in ['i7S', 'i12S', 'i19S']:  
            w1, w2, w3 = 1, 1, 100000
        else:
            w1, w2, w3 = 1, 1, 10

    # prepare model and optimizer
    model = get_model(args.model, args.dataset)
    model.init_weights()
    model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, eps=1e-8, 
                                 betas=(0.9, 0.999))

    # resume from existing or start a new session
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format\
                  (args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch{})".format(args.resume, 
                  checkpoint['epoch']))
            save_path = Path(args.resume)
            args.save_path = save_path.parent
            start_epoch = checkpoint['epoch'] + 1
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.exit()
    else:
        if args.dataset in ['i7S', 'i12S', 'i19S']:
            model_id = "{}-{}-initlr{}-iters{}-bsize{}-aug{}-{}".format(\
                        args.dataset, args.model, args.init_lr, args.n_iter,
                        args.batch_size, int(args.aug), args.train_id)
        else:
            model_id = "{}-{}-{}-initlr{}-iters{}-bsize{}-aug{}-{}".format(\
                        args.dataset, args.scene.replace('/','.'),
                        args.model, args.init_lr, args.n_iter, args.batch_size, 
                        int(args.aug), args.train_id)
        save_path = Path(model_id)
        args.save_path = 'checkpoints'/save_path
        args.save_path.mkdir(parents=True, exist_ok=True)
        start_epoch = 1
    
    # start training
    args.n_epoch = int(np.ceil(args.n_iter * args.batch_size / len(dataset)))

    for epoch in range(start_epoch, args.n_epoch+1):
        lr = adjust_lr(optimizer, args.init_lr, (epoch - 1) 
                       * np.ceil(len(dataset) / args.batch_size), 
                       args.n_iter)
        model.train()
        train_loss_list = []
        coord_loss_list = []
        if args.model == 'hscnet':
            lbl_1_loss_list = []
            lbl_2_loss_list = []
                
        for _, (img, coord, mask, lbl_1, lbl_2, lbl_1_oh, 
                lbl_2_oh) in enumerate(tqdm(trainloader)):

            if mask.sum() == 0:
                continue
            optimizer.zero_grad()

            img = img.cuda()
            coord = coord.cuda()
            mask = mask.cuda()

            if args.model == 'hscnet':
                lbl_1 = lbl_1.cuda()
                lbl_2 = lbl_2.cuda()
                lbl_1_oh = lbl_1_oh.cuda()
                lbl_2_oh = lbl_2_oh.cuda()
                coord_pred, lbl_2_pred, lbl_1_pred = model(img,lbl_1_oh,
                                                           lbl_2_oh)
                lbl_1_loss = cls_loss(lbl_1_pred, lbl_1, mask)
                lbl_2_loss = cls_loss(lbl_2_pred, lbl_2, mask)
                coord_loss = reg_loss(coord_pred, coord, mask)
                train_loss = w3*coord_loss + w1*lbl_1_loss + w2*lbl_2_loss
            else:
                coord_pred = model(img)
                coord_loss = reg_loss(coord_pred, coord, mask)
                train_loss = coord_loss
            
            coord_loss_list.append(coord_loss.item())
            if args.model == 'hscnet':
                lbl_1_loss_list.append(lbl_1_loss.item())
                lbl_2_loss_list.append(lbl_2_loss.item())          
            train_loss_list.append(train_loss.item())
            
            train_loss.backward()
            optimizer.step()

        with open(args.save_path/args.log_summary, 'a') as logfile:
            if args.model == 'hscnet':
                logtt = 'Epoch {}/{} - lr: {} - reg_loss: {} - cls_loss_1: {}' \
                        ' - cls_loss_2: {} - train_loss: {} \n'.format(
                         epoch, args.n_epoch, lr, np.mean(coord_loss_list), 
                         np.mean(lbl_1_loss_list), np.mean(lbl_2_loss_list),
                         np.mean(train_loss_list))
            else:
                logtt = 'Epoch {}/{} - lr: {} - reg_loss: {} - train_loss: {}' \
                        '\n'.format(
                         epoch, args.n_epoch, lr, np.mean(coord_loss_list),
                         np.mean(train_loss_list))
            print(logtt)
            logfile.write(logtt)
        
        if epoch % int(np.floor(args.n_epoch / 5.)) == 0:
            save_state(args.save_path, epoch, model, optimizer)

    save_state(args.save_path, epoch, model, optimizer)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hscnet")
    parser.add_argument('--model', nargs='?', type=str, default='hscnet',
                        choices=('hscnet', 'scrnet'),
                        help='Model to use [\'hscnet, scrnet\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='7S', 
                        choices=('7S', '12S', 'i7S', 'i12S', 'i19S',
                        'Cambridge'), help='Dataset to use')
    parser.add_argument('--scene', nargs='?', type=str, default='heads', 
                        help='Scene')
    parser.add_argument('--n_iter', nargs='?', type=int, default=900000,
                        help='# of the iterations')
    parser.add_argument('--init_lr', nargs='?', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--aug', nargs='?', type=str2bool, default=True,
                        help='w/ or w/o data augmentation')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to saved model to resume from')
    parser.add_argument('--data_path', required=True, type=str, 
                        help='Path to dataset')
    parser.add_argument('--log-summary', default='progress_log_summary.txt', 
                        metavar='PATH',
                        help='txt where to save per-epoch stats')
    parser.add_argument('--train_id', nargs='?', type=str, default='',
                        help='An identifier string')
    args = parser.parse_args()

    if args.dataset == '7S':
        if args.scene not in ['chess', 'heads', 'fire', 'office', 'pumpkin',
                              'redkitchen','stairs']:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset == '12S':
        if args.scene not in ['apt1/kitchen', 'apt1/living', 'apt2/bed',
                              'apt2/kitchen', 'apt2/living', 'apt2/luke', 
                              'office1/gates362', 'office1/gates381', 
                              'office1/lounge', 'office1/manolis',
                              'office2/5a', 'office2/5b']:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset == 'Cambridge':
        if args.scene not in ['GreatCourt', 'KingsCollege', 'OldHospital',
                              'ShopFacade', 'StMarysChurch']:
            print('Selected scene is not valid.')
            sys.exit()

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    train(args)
