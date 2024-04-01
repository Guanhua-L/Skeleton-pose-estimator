'''
$ python3 visualize_skeleton.py /home/jxzhe/MARS/FIA_dataset/data/NoCR/20230517_234352_NoCR_8:2_resnet18_inference/s01_a01_r01.csv && ffmpeg -framerate 10 -pattern_type glob -i '/home/jxzhe/MARS/FIA_dataset/data/NoCR/20230517_234352_NoCR_8:2_resnet18_inference/s01_a01_r01_pngs/*.png' /home/jxzhe/MARS/FIA_dataset/data/NoCR/20230517_234352_NoCR_8:2_resnet18_inference/s01_a01_r01_inf.mp4
python3 visualize_skeleton.py /mnt/data/guanhua/SPE/uncertainty_skeleton/CR/s01_a01_r01.csv
'''

from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt

fps, sec = 10, 3  # 10 fps, 輸出 15 秒

fig = plt.figure(figsize=(8, 4), dpi=300)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


def plot(ax, view, x, y, z):
    ax.cla()
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.set_xlim(-.75, .75)
    ax.set_ylim(-.75, .75)
    ax.set_zlim(-.75 + .2, .75 + .2)
    ax.view_init(*view)  # 調視角

    ax.scatter(x[0], y[0], z[0])
    ax.plot(x[1], y[1], z[1], color='tab:blue')


def main():
    with open(argv[1]) as f:
        for i, frame in enumerate(f):
            points = tuple(map(float, frame.split(',')))

            x, y, z = [[]], [[]], [[]]
            for j in range(0, len(points), 3):
                x[0].append(points[j])
                y[0].append(points[j + 2])
                z[0].append(points[j + 1])

            x.append((x[0][11], x[0][5], *x[0][9:0:-2], *x[0][2:11:2], x[0][6], x[0][12]))
            y.append((y[0][11], y[0][5], *y[0][9:0:-2], *y[0][2:11:2], y[0][6], y[0][12]))
            z.append((z[0][11], z[0][5], *z[0][9:0:-2], *z[0][2:11:2], z[0][6], z[0][12]))

            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)

            plot(ax1, (0, 270), x, y, z)
            plot(ax2, (0, 0), x, y, z)

            fig.suptitle(f'{i:03}', fontsize=14)
            
            path = Path(argv[1])
            (path.parent / f'{path.stem}_pngs').mkdir(mode=0o777, parents=True, exist_ok=True)
            plt.savefig(path.parent / f'{path.stem}_pngs/{i:03}.png', bbox_inches='tight', pad_inches=0, dpi=300)
            print(f'{i:03}')

            # if i == fps * sec - 1:
            #     break


if __name__ == '__main__':
    main()
