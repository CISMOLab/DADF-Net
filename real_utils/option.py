import argparse
import yaml

trainSetFile = open("/home/graduate/liulei/DADF-Net/real_utils/trainSetting.yaml", "r", encoding="utf-8")
trainSet = trainSetFile.read()
trainSetFile.close()
set = yaml.load(trainSet, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default=set['gpu_id'])

# Data specifications
parser.add_argument('--data_root', type=str, default=set['data_root'], help='dataset directory')
parser.add_argument('--data_path_CAVE', default=set['data_path_CAVE'], type=str,
                        help='path of data')
parser.add_argument('--data_path_KAIST', default=set['data_path_KAIST'], type=str,
                    help='path of data')
parser.add_argument('--mask_path', default=set['mask_path'], type=str,
                    help='path of mask')

# Saving specifications
parser.add_argument('--outf', type=str, default=set['outf'], help='saving_path')

# Model specifications
parser.add_argument("--input_setting", type=str, default=set['input_setting'],
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default=set['input_mask'],
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask


# Training specifications
parser.add_argument("--loss_type", type=str, default=set['loss_type'])
parser.add_argument("--size", default=set['size'], type=int, help='cropped patch size')
parser.add_argument("--epoch_sam_num", default=set['epoch_sam_num'], type=int, help='total number of trainset')
parser.add_argument("--seed", default=set['seed'], type=int, help='Random_seed')
parser.add_argument('--batch_size', type=int, default=set['batch_size'], help='the number of HSIs per batch')
parser.add_argument("--isTrain", default=set['isTrain'], type=bool, help='train or test')
parser.add_argument("--max_epoch", type=int, default=set['max_epoch'], help='total epoch')
parser.add_argument("--scheduler", type=str, default=set['scheduler'], help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=set['milestones'], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=set['gamma'], help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=set['learning_rate'])
parser.add_argument("--RESUME", type=bool, default=set['RESUME'])
parser.add_argument("--re_path", type=list, default=set['re_path'],help='[model_path,result_path]')
opt = parser.parse_args()

ConfigFile = open("/home/graduate/liulei/DADF-Net/real_utils/config.yaml", "r", encoding="utf-8")
MODELCONFIG = ConfigFile.read()
ConfigFile.close()
config = yaml.load(MODELCONFIG, Loader=yaml.FullLoader)

opt.trainset_num = 20000 // ((opt.size // 96) ** 2)
opt.epoch_sam_num = opt.trainset_num
for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False