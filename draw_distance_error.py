import csv
import numpy as np
import matplotlib.pyplot as plt

def draw2(categories, values, values2, color, color2, title, target):
    bars = plt.bar(categories, values, color=color, width=-0.4, align='edge', label='CR')
    bars2 = plt.bar(categories, values2, color=color2, width=0.4, align='edge', label='NoCR')
    # plt.bar(categories, values, color=color, width=0.4, align='edge')
    # plt.bar(categories, values2, color=color2, width=0.4)

    # plt.xlabel('type')
    plt.ylabel('distance error (cm)')
    # plt.title(title)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('orange', 'dodgerblue', 'gold', 'lightblue')]
    plt.legend(legend_handles, ['Baseline', 'Baseline', 'Transformer', 'Transformer'], bbox_to_anchor=(1.04, 0.25), loc='upper left', fontsize=12, edgecolor='#000')
    plt.tight_layout()
    plt.savefig(target)
    plt.cla()
    # plt.show()

path_transformer = ["./result/single_transformer/20240606_232614_CR_8:2_fold1_resnet34/result.csv",
                    "./result/single_transformer/20240607_004813_CR_test1_resnet34/result.csv",
                    "./result/single_transformer/20240607_022833_CR_test2_resnet34/result.csv",
                    "./result/single_transformer/20240606_035323_CR_test3_resnet34/result.csv"]

path_baseline = ["./result/20240604_201753_CR_8:2_fold5_resnet34/result.csv",
                 "./result/20240605_160635_CR_test1_resnet34/result.csv",
                 "./result/20240605_211732_CR_test2_resnet34/result.csv",
                 "./result/20240606_022421_CR_test3_resnet34/result.csv"]
print(len(path_transformer)," ", len(path_baseline))
for index in range(0,len(path_transformer)-1):
    print(f'{index}: {path_transformer[index]} , {path_baseline[index]}')
    test_distance_error = []
    test_distance_error_baseline = []

    train_distance_error = []
    train_distance_error_baseline = []

    with open(path_transformer[index], newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for j, row in enumerate(rows):
            test_distance_error.append(float(row['test_distance_mean']))
            train_distance_error.append(float(row['train_distance']))
            # print(j, ' ', row['test_distance_mean'])

    with open(path_baseline[index], newline='') as csvfile2:
        rows = csv.DictReader(csvfile2)
        for j, row in enumerate(rows):
            test_distance_error_baseline.append(float(row['test_distance_mean']))
            train_distance_error_baseline.append(float(row['train_distance']))
            # print(j, ' ', row['test_distance_mean'])


    x = []
    for i in range(1, 31):
        x.append(i)
    # print(test_distance_error)

    plt.plot(x[:len(test_distance_error)], test_distance_error, color='orange', marker='o', linestyle='--', linewidth=1, markersize=4)
    plt.plot(x[:len(test_distance_error_baseline)], test_distance_error_baseline, color='b', marker='o', linestyle='--', linewidth=1, markersize=4)
    if index == 0:
        plt.legend(['Transformer', 'Baseline'], bbox_to_anchor=(0.5,0), loc='lower right')
    else:
        plt.legend(['Transformer', 'Baseline'], bbox_to_anchor=(1,0), loc='lower right')
    for i, x in enumerate(test_distance_error):
        if i % 29 == 0:
            plt.text(i+0.8, x+0.1, f'{x:.02f}', fontsize=10, color='black', verticalalignment='baseline', horizontalalignment='center')
    for i, x in enumerate(test_distance_error_baseline):
        if i % 29 == 0:
            plt.text(i+0.8, x+0.1, f'{x:.02f}', fontsize=10, color='black', verticalalignment='baseline', horizontalalignment='center')
    plt.xlabel('epoch')
    plt.ylabel('distance error (cm)')
    plt.tight_layout()

    plt.savefig(f'./figure/test_distance_error_{index}.png')
    plt.cla()

    categories = ['train', 'test']
    colors = ['orange', 'dodgerblue', 'orange', 'dodgerblue']
    colors2 = ['gold', 'lightblue', 'gold', 'lightblue']
    value = [train_distance_error_baseline[-1], test_distance_error_baseline[-1]]
    value2 = [train_distance_error[-1], test_distance_error[-1]]
    draw2(categories, value, value2, colors, colors2, f'{index}', f'./figure/bar_test_distance_error_{index}.png')