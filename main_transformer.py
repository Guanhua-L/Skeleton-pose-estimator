import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# from train_org import Trainer
from train_single_transformer import Trainer
# from train import Trainer


def init_logger():
    logging.config.fileConfig(
        fname='configs/log_config.ini',
        defaults={
            'logfilename_s': f'log/{datetime.now().strftime("%Y%m%d_%H%M%S.log")}',
            'logfilename_c': f'log/{datetime.now().strftime("%Y%m%d_%H%M%S_client.log")}'
        }
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info('New training--------------------\n')
    print(f'[{datetime.now()}] Start new training')


def init_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, default='resnet18', required=True, help='Select model (MARS or resnet18)')
    parser.add_argument('-g', '--gpu', type=int, default=0, required=True, help='Select GPU (0~3)')
    parser.add_argument('-c', '--cr', type=str, default='NoCR', required=True, help='Select CR or NoCR')
    parser.add_argument('-t', '--test', type=int, default=0, required=True, help='Select the testing subject (1~24), 0 for 8:2 random split')
    parser.add_argument('-f', '--fold', type=int, default=1, required=False, help='Select the fold (1~5 for 5x cv)')
    args = parser.parse_args()
    return args


def init_path(args: argparse.Namespace) -> Path:
    result_path = Path(f'result/single_transformer/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.cr}_{f"test{args.test}" if args.test else f"8:2_fold{args.fold}"}_{args.model}')
    # result_path = Path(f'/mnt/nas/FIA/MARS_result/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.cr}_{f"test{args.test}" if args.test else "8:2"}_{args.model}')
    (result_path / 'checkpoints').mkdir(mode=0o777, parents=True)
    pd.read_csv('configs/training_config_CNNLSTM_transformer.csv', index_col=0).loc[[args.model]].to_csv(result_path / 'training_config.csv')
    print(f'Results will be saved in {result_path}')
    return result_path


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    # init_logger()
    set_all_seeds(42)
    args = init_parser()
    result_path = init_path(args)
    Trainer(args, result_path).train()

if __name__ == '__main__':
    main()
