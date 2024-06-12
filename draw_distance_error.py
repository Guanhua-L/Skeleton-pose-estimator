import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontsize = (31, 29, 27, 29)

def draw2(categories, values, values2, color, color2, title, target):
    bars = plt.bar(categories, values, color=color, width=-0.4, align='edge', label='CR')
    bars2 = plt.bar(categories, values2, color=color2, width=0.4, align='edge', label='NoCR')
    # plt.bar(categories, values, color=color, width=0.4, align='edge')
    # plt.bar(categories, values2, color=color2, width=0.4)

    # plt.xlabel('Train or Test', fontsize=fontsize[3])
    plt.ylabel('distance error (cm)', fontsize=fontsize[3])
    plt.yticks(fontsize=fontsize[3])
    plt.xticks(fontsize=fontsize[3])
    # plt.title(title)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize[3])
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize[3])
    
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('dodgerblue', 'orange')]
    # plt.legend(legend_handles, ['Baseline', 'Baseline', 'Transformer', 'Transformer'], bbox_to_anchor=(1.04, 0.25), loc='upper left', fontsize=fontsize[3], edgecolor='#000')
    plt.legend(legend_handles, ['Baseline', 'Transformer'], loc='lower right', fontsize=12, edgecolor='#000')
    plt.tight_layout()
    plt.savefig(target, dpi=300, bbox_inches='tight')
    plt.savefig(f'{target[:-3]}eps', bbox_inches='tight')
    plt.cla()
    # plt.show()

path_transformer = ["./transformer_result/20240606_232614_CR_8:2_fold1_resnet34/result.csv",
                    "./transformer_result/20240607_004813_CR_test1_resnet34/result.csv",]

path_baseline = ["./baseline/20240604_201753_CR_8:2_fold5_resnet34/result.csv",
                 "./baseline/20240605_160635_CR_test1_resnet34/result.csv",
                 "./baseline/20240605_211732_CR_test2_resnet34/result.csv",
                 "./baseline/20240606_022421_CR_test3_resnet34/result.csv",
                 "./baseline/20240606_073405_CR_test4_resnet34/result.csv",
                 "./baseline/20240606_124400_CR_test5_resnet34/result.csv",
                 "./baseline/20240606_175420_CR_test6_resnet34/result.csv",
                 "./baseline/20240606_230547_CR_test7_resnet34/result.csv",
                 "./baseline/20240607_041652_CR_test8_resnet34/result.csv",
                 "./baseline/20240607_092708_CR_test9_resnet34/result.csv",
                 "./baseline/20240607_143755_CR_test10_resnet34/result.csv",
                 "./baseline/20240607_194728_CR_test11_resnet34/result.csv",
                 "./baseline/20240608_005658_CR_test12_resnet34/result.csv",
                 "./baseline/20240608_060520_CR_test13_resnet34/result.csv",
                 "./baseline/20240608_111354_CR_test14_resnet34/result.csv",
                 "./baseline/20240608_162208_CR_test15_resnet34/result.csv",
                 "./baseline/20240608_213034_CR_test16_resnet34/result.csv",
                 "./baseline/20240609_023906_CR_test17_resnet34/result.csv",
                 "./baseline/20240609_074743_CR_test18_resnet34/result.csv",
                 "./baseline/20240609_125558_CR_test19_resnet34/result.csv",
                 "./baseline/20240609_180357_CR_test20_resnet34/result.csv",
                 "./baseline/20240609_231231_CR_test21_resnet34/result.csv",
                 "./baseline/20240610_042055_CR_test22_resnet34/result.csv",
                 "./baseline/20240610_092859_CR_test23_resnet34/result.csv",
                 "./baseline/20240610_143805_CR_test24_resnet34/result.csv"]
print(len(path_transformer)," ", len(path_baseline))

length = min(len(path_baseline), len(path_transformer))
for index in range(0,length):
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
            plt.text(i+0.8, x+0.1, f'{x:.02f}', fontsize=16, color='black', verticalalignment='baseline', horizontalalignment='center')
    for i, x in enumerate(test_distance_error_baseline):
        if i % 29 == 0:
            plt.text(i+0.8, x+0.1, f'{x:.02f}', fontsize=16, color='black', verticalalignment='baseline', horizontalalignment='center')
    plt.xlabel('epoch', fontsize=fontsize[3])
    plt.ylabel('distance error (cm)', fontsize=fontsize[3])
    plt.yticks(fontsize=fontsize[3])
    plt.xticks(fontsize=fontsize[3])
    plt.tight_layout()

    plt.savefig(f'./figure/test_distance_error_{index}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./figure/test_distance_error_{index}.eps', bbox_inches='tight')
    plt.cla()

    categories = ['train', 'test']
    # colors = ['dodgerblue', 'lightblue', 'dodgerblue', 'lightblue']
    # colors2 = ['orange', 'gold', 'orange', 'gold']
    colors = ['dodgerblue', 'dodgerblue', 'dodgerblue', 'dodgerblue']
    colors2 = ['orange', 'orange', 'orange', 'orange']
    value = [train_distance_error_baseline[-1], test_distance_error_baseline[-1]]
    value2 = [train_distance_error[-1], test_distance_error[-1]]
    draw2(categories, value, value2, colors, colors2, f'{index}', f'./figure/bar_test_distance_error_{index}.png')