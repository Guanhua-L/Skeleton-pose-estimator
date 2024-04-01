# %%
import pandas as pd
import torch

from util import get_distance

pred = torch.from_numpy(pd.read_csv('/home/jxzhe/MARS/FIA_dataset/data/NoCR/20230517_234352_NoCR_8:2_resnet18_inference/s01_a01_r01.csv').to_numpy()[:-2])
labels = torch.from_numpy(pd.read_csv('/home/jxzhe/MARS/FIA_dataset/labels/s01_a01_r01.csv').to_numpy())
print(get_distance(pred, labels) * 100)