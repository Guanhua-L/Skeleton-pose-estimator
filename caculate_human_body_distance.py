import csv
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    for path in sorted(Path('./FIA_dataset/labels').glob('*.csv')):
        print(path)
        with open(path, newline='') as f:
            datas = csv.reader(f, delimiter=' ')
            for data in datas:
                print(data)
                exit()