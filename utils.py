from __future__ import division

import os
import torch

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def adjust_lr(optimizer, init_lr, c_iter, n_iter):
    lr = init_lr * (0.5 ** ((c_iter + 200000 - n_iter) // 50000 + 1 if (c_iter 
         + 200000 - n_iter) >= 0 else 0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr    

def save_state(savepath, epoch, model, optimizer):
    state = {'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()}
    filepath = os.path.join(savepath, 'model.pkl')
    torch.save(state, filepath)