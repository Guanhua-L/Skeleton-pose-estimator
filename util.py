from pathlib import Path

import pandas as pd
import pynvml
import torch


def get_config_2DCNN(result_path: Path, model: str):
    '''
        Returns:
            learning_rate, batch_size, epoch, weight_decay, lr_decay
    '''
    learning_rate, batch_size, epoch, weight_decay, lr_decay = pd.read_csv(result_path / 'training_config.csv', index_col=0).loc[model]
    return learning_rate, int(batch_size), int(epoch), weight_decay, lr_decay

def get_config_CNNLSTM(result_path: Path, model: str):
    '''
        Returns:
            learning_rate, batch_size, epoch, weight_decay, lr_decay, input_size, hidden_size, num_layers, dropout, bidirectional, window
    '''
    learning_rate, batch_size, epoch, weight_decay, lr_decay, input_size, hidden_size, num_layers, dropout, bidirectional, window = pd.read_csv(result_path / 'training_config.csv', index_col=0).loc[model]
    return learning_rate, int(batch_size), int(epoch), weight_decay, lr_decay, int(input_size), int(hidden_size), int(num_layers), dropout, bool(bidirectional), int(window)


def is_gpu_available(gpu: int) -> bool:
    pynvml.nvmlInit()
    num_GPUs = pynvml.nvmlDeviceGetCount()
    if gpu >= num_GPUs or gpu < 0:
        return False
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free / (1024 * 1024) > 4500


def init_device(gpu: int) -> str:
    if gpu == -1:
        return 'cpu'
    if is_gpu_available(gpu):
        return f'cuda:{gpu}'
    raise Exception(f'GPU {gpu} is unavailable!')


def get_mae(pred: torch.Tensor, labels: torch.Tensor):  # shape = (batch_size, 39)
    mae_entire_batch = torch.zeros((pred.shape[0], pred.shape[1] // 3))
    for i in range(0, pred.shape[1], 3):
        mae_entire_batch[:, i // 3] = torch.linalg.norm(pred[:, i:i + 3] - labels[:, i:i + 3], ord=1, dim=1) / 3
    # return mae_entire_batch.mean(0)
    return mae_entire_batch


def get_distance(pred: torch.Tensor, labels: torch.Tensor):  # shape = (batch_size, 39)
    distance_entire_batch = torch.zeros((pred.shape[0], pred.shape[1] // 3))
    for i in range(0, pred.shape[1], 3):
        distance_entire_batch[:, i // 3] = torch.linalg.norm(pred[:, i:i + 3] - labels[:, i:i + 3], dim=1)
    # return distance_entire_batch.mean(0)
    return distance_entire_batch
