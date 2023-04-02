from models.NetMaker import model_bulider
from utils.utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option.option import opt,config
import torch.nn.functional as F
from tqdm import tqdm

torch.cuda.set_device(int(opt.gpu_id))
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

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

mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)

train_set = LoadTraining(opt.data_path)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
if opt.loss_type=='L2':
    mse = torch.nn.MSELoss().cuda()

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    train_logger = tqdm(range(batch_num))
    for i in train_logger:
        gt_batch = shuffle_crop(train_set, opt.batch_size)
        gt = Variable(gt_batch).cuda().float()
        input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)
        optimizer.zero_grad()
        model_out = model(input_meas,input_mask_train)
        if opt.loss_type=='L2' and config['Multiscale']:
            label_img2 = F.interpolate(gt, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(gt, scale_factor=0.25, mode='bilinear')            
            loss1 = torch.sqrt(mse(model_out[0], label_img4))
            loss2 = torch.sqrt(mse(model_out[1], label_img2))
            loss3 = torch.sqrt(mse(model_out[2], gt))
            loss = loss1+loss2+loss3
        elif opt.loss_type=='L2' and not config['Multiscale']:
            loss = torch.sqrt(mse(model_out, gt))
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        train_logger.set_description(desc='[epoch: %d][lr: %.6f][loss: %.6f][mean_loss: %.6f]'%(epoch,scheduler.get_last_lr()[0],loss,epoch_loss / (i+1)))
    end = time.time()
    logger.info("===> Epoch {} Complete: lr:{:.6f} Avg. Loss: {:.6f} time: {:.2f}".format(epoch,scheduler.get_last_lr()[0],epoch_loss / batch_num, (end - begin)))
    return 0

def main():
    start_epoch = 0
    if opt.RESUME:
        path_checkpoint = os.path.join(model_path, 'mycheckpoint.pth')
        recheckpoint = torch.load(path_checkpoint)

        model.load_state_dict(recheckpoint['net'])

        optimizer.load_state_dict(recheckpoint['optimizer'])
        start_epoch = recheckpoint['epoch']
    for epoch in range(start_epoch+1, opt.max_epoch + 1):
        train(epoch, logger)
        scheduler.step()
        mycheckpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(mycheckpoint, os.path.join(model_path, 'mycheckpoint.pth'))
        checkpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    main()