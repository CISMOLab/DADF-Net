from models.NetMaker import model_bulider

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import scipy.io as scio
import time
import os
import numpy as np
import datetime
from tqdm import tqdm

from real_utils.dataset import *
from real_utils.utils import *
from real_utils.option import opt,config
# import pytorch_warmup as warmup


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

torch.cuda.set_device(int(opt.gpu_id))
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')


# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + str(opt.epoch_sam_num) + '/' + date_time + '/result/'
model_path = opt.outf + str(opt.epoch_sam_num)  + '/' + date_time  + '/model/'

if opt.RESUME:
    model_path = opt.re_path[0]
    result_path = opt.re_path[1]

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

logger = gen_log(model_path)
logger.info("\n trainSetting:{}\n model config:{}\n".format(opt, config))

model = model_bulider(config).cuda()

CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30)

# model



# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)

if opt.loss_type=='L2':
    mse = torch.nn.MSELoss().cuda()
elif opt.loss_type=='L1':
    char_loss = L1_Charbonnier_loss().cuda()

def main():
    print(opt,config)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    ## pipline of training
    for epoch in range(1, opt.max_epoch+1):
        model.train()
        Dataset = dataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(Dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)

        # scheduler.step()
        epoch_loss = 0

        start_time = time.time()
        train_logger = tqdm(enumerate(loader_train))
        for i, (input, label, Mask, Phi, Phi_s) in train_logger:
            input, label, Phi, Phi_s = Variable(input), Variable(label), Variable(Phi), Variable(Phi_s)
            input, label, Phi, Phi_s = input.cuda(), label.cuda(), Phi.cuda(), Phi_s.cuda()

            input_mask = init_mask(Mask, Phi, Phi_s, opt.input_mask).cuda()
            
            input = y2x(input)

            model_out = model(input, input_mask)

            # if opt.loss_type=='L2':
            #     loss = torch.sqrt(mse(model_out, label))
            # elif opt.loss_type=='L1':
            #     loss = char_loss(model_out, label)


            if opt.loss_type=='L2' and config['Multiscale']:
                label_img2 = F.interpolate(label, scale_factor=0.5, mode='bilinear')
                label_img4 = F.interpolate(label, scale_factor=0.25, mode='bilinear')            
                loss1 = torch.sqrt(mse(model_out[0], label_img4))
                loss2 = torch.sqrt(mse(model_out[1], label_img2))
                loss3 = torch.sqrt(mse(model_out[2], label))
                loss = loss1+loss2+loss3
            # elif opt.loss_type=='mix':
            #     label_img2 = F.interpolate(gt, scale_factor=0.5, mode='bilinear')
            #     label_img4 = F.interpolate(gt, scale_factor=0.25, mode='bilinear')            
            #     loss1 = torch.sqrt(mse(model_out[0], label_img4)) + 0.1*edge(model_out[0], label_img4)
            #     loss2 = torch.sqrt(mse(model_out[1], label_img2)) + 0.1*edge(model_out[1], label_img2)
            #     loss3 = torch.sqrt(mse(model_out[2], gt)) + 0.1*edge(model_out[2], gt)
            #     loss = loss1+loss2+loss3
            elif opt.loss_type=='L2' and not config['Multiscale']:
                loss = torch.sqrt(mse(model_out, label))
            elif opt.loss_type=='L1':
                loss = char_loss(model_out, label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_logger.set_description(desc='[epoch: %d][lr: %.6f][loss: %.6f][mean_loss: %.6f]'%(epoch,scheduler.get_last_lr()[0],loss,epoch_loss / ((i + 1) * opt.batch_size)))
            
            # if i % (1000) == 0:
            #     print('%4d %4d / %4d loss = %.10f time = %s' % (
            #         epoch + 1, i, len(Dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
            #         datetime.datetime.now()))
        scheduler.step()
        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch, epoch_loss / len(Dataset), elapsed_time))
        logger.info("===> Epoch {} Complete: lr:{:.6f} Avg. Loss: {:.6f} time: {:.2f}".format(epoch,scheduler.get_last_lr()[0],epoch_loss / len(Dataset), elapsed_time))
        torch.save(model, os.path.join(model_path, 'model_%03d.pth' % (epoch)))
        

if __name__ == '__main__':
    main()