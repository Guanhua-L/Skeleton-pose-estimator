import argparse
from pathlib import Path
from shutil import copy2, rmtree

import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r', '--result_path', type=str, required=True, help='Path to the "result" folder')
args = parser.parse_args()
result_path = Path(args.result_path)
average = [0] * 24
for path in sorted(result_path.glob('**/result.csv')):
    '''Move Best Epoch Outside'''
    # if len(list(path.parent.glob('*.pt'))):
    #     continue
    df = pd.read_csv(path)
    if 'test_distance' in df:
        best_epoch = int(df[df.test_distance == df.test_distance.min()]['Unnamed: 0'])
    elif 'test_distance_mean' in df:
        best_epoch = int(df[df.test_distance_mean == df.test_distance_mean.min()]['Unnamed: 0'])
    else:
        raise f'Something\'s fucking wrong in {path}'
    average[int(path.parent.stem.split('_')[-2][4:]) - 1] = df.iloc[best_epoch - 1]['test_distance_mean']
    # copy2(path.parent / f'checkpoints/{best_epoch}.pt', path.parent)

    '''Delete Checkpoints Folder'''
    # if (path.parent / 'checkpoints').exists() and len(list(path.parent.glob('*.pt'))):
    #     print(path.parent.stem)
    #     rmtree(path.parent / 'checkpoints')
print(sum(average) / 24)