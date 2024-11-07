import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
# from utils import DiceLoss

from torchvision import transforms
from icecream import ic
from datetime import datetime
from datasets.data import build_data
from einops import rearrange

import matplotlib.pyplot as plt

def calc_loss(outputs, low_res_label_batch, ce_loss):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    return loss_ce

def trainer_run(args, model, snapshot_path, multimask_output, phase):
    
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=10)
    
    dataset_train, dataset_valid = build_data(args)
    
    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if not os.path.exists('./training_log'):
        os.mkdir('./training_log')
    logging.basicConfig(filename= './training_log/' + args.output.split('/')[-1] + '_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    print("The length of train set is: {}".format(len(dataset_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if phase == "train":
        trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                worker_init_fn=worker_init_fn, drop_last=True)
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
        model.train()
        
        if args.warmup:
            b_lr = base_lr / args.warmup_period
        else:
            b_lr = base_lr
        if args.AdamW:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001) 
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        writer = SummaryWriter(snapshot_path + '/log')
        iter_num = 0
        max_epoch = args.max_epochs
        stop_epoch = args.stop_epoch
        max_iterations = args.max_epochs * len(trainloader)
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
        
        iterator = tqdm(range(max_epoch), ncols=70)

        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(trainloader):
                image_batch, wave_let, label_batch = sampled_batch['image'], sampled_batch['wavelet'], sampled_batch['label']
                image_batch, wave_let, label_batch = image_batch.cuda().float(), wave_let.cuda().float(), label_batch.cuda()
                image_batch = rearrange(image_batch, 'b c d h w -> (b d) c h w ')
                wave_let = rearrange(wave_let, 'b d h w -> (b d) h w ')
                
                if args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                        # outputs = model(image_batch, multimask_output, args.img_size)
                        outputs = model(image_batch, wave_let, multimask_output, args.img_size)
                        label_batch = rearrange(label_batch[:,0,:,:,:], 'b d h w-> (b d) h w ')
                        loss  = calc_loss(outputs, label_batch, ce_loss)
                        
                        result = outputs['low_res_logits'].argmax(dim=1)
                        
                        # if epoch_num % 5 == 0 and i_batch % 50 == 1:
                        #     # Overlay here.
                        #     from utils.EchoLib.IO_handler_delete.infer_utils import draw_contour
                        #     image_batch = rearrange(image_batch, '(b d) c h w -> b d c h w', b=2, d=10)
                        #     label_batch = rearrange(label_batch, '(b d) h w -> b d h w', b=2, d=10)
                        #     result = rearrange(result, '(b d) h w -> b d h w', b=2, d=10)
                            
                        #     for bidx in range(2):
                        #         for fr in range(10):
                        #             fr = int(fr)
                        #             image = image_batch[bidx][0]
                        #             label = label_batch[bidx][fr]
                        #             pred = result[bidx][fr]
                        #             epoch_name =  os.path.basename(args.adapt_ckpt)
                        #             draw_contour(image, label, pred, 4, epoch_name)
                        #     plt.imshow(result[0].cpu().detach().numpy())
                        #     plt.savefig('test.png')
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                if args.warmup and iter_num < args.warmup_period:
                    lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                else:
                    if args.warmup:
                        shift_iter = iter_num - args.warmup_period
                        assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    else:
                        shift_iter = iter_num
                    lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)

                logging.info('iteration %d : loss : %f ' % (iter_num, loss.item()))

            save_interval = 10
            if (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                try:
                    model.save_parameters(save_mode_path)
                except:
                    model.module.save_parameters(save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                try:
                    model.save_parameters(save_mode_path)
                except:
                    model.module.save_parameters(save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                iterator.close()
                break
    else:        
        val_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                worker_init_fn=worker_init_fn, drop_last=True)
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
        model.eval()

        for i_batch, sampled_batch in enumerate(val_loader):
            image_batch, wave_let, label_batch = sampled_batch['image'], sampled_batch['wavelet'], sampled_batch['label'] 
            image_batch, wave_let, label_batch = image_batch.cuda().float(), wave_let.cuda().float(), label_batch.cuda()
            image_batch = rearrange(image_batch, 'b c d h w -> (b d) c h w ')
            wave_let = rearrange(wave_let, 'b d h w -> (b d) h w ')
            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    # outputs = model(image_batch, multimask_output, args.img_size)
                    outputs = model(image_batch, wave_let, multimask_output, args.img_size)
                    label_batch = rearrange(label_batch[:,0,:,:,:], 'b d h w-> (b d) h w ')
                    result = outputs['low_res_logits'].argmax(dim=1)
                    
                    #Visualization
                    
                    
    writer.close()
    return "Training Finished!"