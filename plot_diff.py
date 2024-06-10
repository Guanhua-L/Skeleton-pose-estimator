import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('/mnt/data/guanhua/SPE/baseline/8:2/result.csv')
df2 = pd.read_csv('/mnt/data/guanhua/SPE/result/20240529_135611_CR_8:2_fold1_resnet34/result.csv') 
df2 = pd.read_csv('/mnt/data/guanhua/SPE/result/20240530_132352_CR_8:2_fold1_resnet34/result.csv') #/mnt/data/guanhua/SPE/result/20240530_132352_CR_8:2_fold1_resnet34/result.csv

test_distance_mean1 = df1['test_distance_mean']
test_distance_mean2 = df2['test_distance_mean']

plt.plot(test_distance_mean1, color='blue', label='Baseline')
plt.plot(test_distance_mean2, color='orange', label='Transformer')

plt.xlabel('Epoch')
plt.ylabel('Distance Error')

plt.legend()

plt.text(0, test_distance_mean1.iloc[0], f"{test_distance_mean1.iloc[0]:.2f}", fontsize=10, ha='center', va='bottom')
plt.text(len(test_distance_mean1)-1, test_distance_mean1.iloc[-1], f"{test_distance_mean1.iloc[-1]:.2f}", fontsize=10, ha='center', va='bottom')
plt.text(0, test_distance_mean2.iloc[0], f"{test_distance_mean2.iloc[0]:.2f}", fontsize=10, ha='center', va='bottom')
plt.text(len(test_distance_mean2)-1, test_distance_mean2.iloc[-1], f"{test_distance_mean2.iloc[-1]:.2f}", fontsize=10, ha='center', va='bottom')

plt.savefig('diff.png')