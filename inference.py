'''
python3 inference.py 0 /home/jxzhe/MARS/result/20230511_022730_NoCR_test7/checkpoints/144.pt /home/jxzhe/MARS/FIA_dataset/data/NoCR/s01_a01_r01.npy

python3 visualize_skeleton.py /home/jxzhe/MARS/FIA_dataset/data/NoCR/s01_a01_r01_inference.csv

ffmpeg -framerate 10 -pattern_type glob -i '/home/jxzhe/MARS/FIA_dataset/data/NoCR/s01_a01_r01_inference_pngs/*.png' /home/jxzhe/MARS/FIA_dataset/data/NoCR/s01_a01_r01_inference.mp4 -y
'''

'''
python3 inference.py 0 /mnt/data/guanhua/SPE/result/20231018_103805_CR_8:2_fold1_resnet34/checkpoints/30.pt
                        /mnt/data/guanhua/SPE/skeleton_driver/20231024_body_mmw.npy

python3 visualize_skeleton.py /mnt/data/guanhua/SPE/uncertainty_skeleton/CR/s01_a01_r01.csv


'''

from pathlib import Path
from sys import argv, exit
from time import time

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18

from model import MARS, CNN_LSTM
from torchvision.models import (alexnet, googlenet, inception_v3, resnet18, resnet34, resnet50)
from util import get_config_2DCNN, get_config_CNNLSTM, get_distance, get_mae, init_device

device = f'cuda:{argv[1]}'
# model = CNN_LSTM(resnet34, 39, 128, 1, 1, ).to(device)
lr, batch_size, epoch, weight_decay, lr_decay, input_size, hidden_size, num_layers, dropout, bidirectional, window = get_config_CNNLSTM(Path('/mnt/ssd1/SPE/result/20231018_155856_NoCR_8:2_fold1_resnet34'), 'resnet34')
model = CNN_LSTM(resnet34, input_size, hidden_size, num_layers, 39, dropout, bidirectional, device, window).to(device)

model.load_state_dict(torch.load(argv[2]))  # path to model checkpoint
model.eval()
for path in sorted(Path('/mnt/data/guanhua/SPE/spe_skeleton/NoCR').glob('*.npy')):
    data = torch.from_numpy(np.load(path))  # path to inference data
    result = np.zeros((data.shape[0] - window + 1, 39))
    # print(data.shape)
    # print(result.shape)
    with torch.no_grad():
        for frame in range(9, data.shape[0]):

            data_batch = data[None, frame - 9:frame].to(device)

            start_time = time()
            pred = model(data_batch)
            end_time = time()
            inference_time = (end_time - start_time) * 1000
            print(f'{path.stem}: {inference_time = } ms')
            result[frame - 9] = pred.cpu().numpy()
            # exit()

        df = pd.DataFrame(result)
        df.to_csv(Path('/mnt/data/guanhua/SPE/uncertainty_skeleton/CR') / f'{path.stem}.csv', header=None, index=None)

# for path in sorted(Path('/home/jxzhe/MARS/FIA_dataset/data/NoCR').glob('*.npy')):
#     # print(path)
#     # continue
#     data = torch.from_numpy(np.load(path))  # path to inference data
#     with torch.no_grad():
#         data = data.to(device)

#         start_time = time()
#         pred = model(data)
#         end_time = time()
#         inference_time = (end_time - start_time) * 1000
#         print(f'{path.stem}: {inference_time = } ms')

#         df = pd.DataFrame(pred.cpu().numpy())
#         df.to_csv(path.parent / '20230517_234352_NoCR_8:2_resnet18_inference' / f'{path.stem}.csv', header=None, index=None)
