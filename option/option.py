import argparse
import yaml

trainSetFile = open("./option/trainSetting.yaml", "r", encoding="utf-8")
trainSet = trainSetFile.read()
trainSetFile.close()
set = yaml.load(trainSet, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser(description="Reconstruction")
parser.add_argument('--template', default='Reconstruction')

parser.add_argument("--gpu_id", type=str, default=set['gpu_id'])

parser.add_argument('--data_root', type=str, default=set['data_root'])

parser.add_argument('--outf', type=str, default=set['outf'], help='saving_path')

parser.add_argument("--input_setting", type=str, default=set['input_setting'])
parser.add_argument("--input_mask", type=str, default=set['input_mask'])  

parser.add_argument('--batch_size', type=int, default=set['batch_size'])
parser.add_argument("--max_epoch", type=int, default=set['max_epoch'])
parser.add_argument("--loss_type", type=str, default=set['loss_type'])
parser.add_argument("--scheduler", type=str, default=set['scheduler'])
parser.add_argument("--milestones", type=int, default=set['milestones'])
parser.add_argument("--gamma", type=float, default=set['gamma'])
parser.add_argument("--epoch_sam_num", type=int, default=set['epoch_sam_num'])
parser.add_argument("--learning_rate", type=float, default=set['learning_rate'])
parser.add_argument("--RESUME", type=bool, default=set['RESUME'])
parser.add_argument("--re_path", type=list, default=set['re_path'],help='[model_path,result_path]')
opt = parser.parse_args()

ConfigFile = open("./option/config.yaml", "r", encoding="utf-8")
MODELCONFIG = ConfigFile.read()
ConfigFile.close()
config = yaml.load(MODELCONFIG, Loader=yaml.FullLoader)
# dataset
opt.data_path = f"{opt.data_root}/cave_1024_28/"
opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False