import numpy as np
import math

list = [72.43, 73.01, 73.12, 72.99, 72.88]

list = np.array(list)

mean = list.mean()
print(mean)


exp_var = 0.0
for i in list:
    exp_var += (i - mean)**2
exp_var /= len(list)-1
print(exp_var)

exp_var_mean = exp_var/len(list)
print(exp_var_mean)

print(math.sqrt(exp_var_mean))