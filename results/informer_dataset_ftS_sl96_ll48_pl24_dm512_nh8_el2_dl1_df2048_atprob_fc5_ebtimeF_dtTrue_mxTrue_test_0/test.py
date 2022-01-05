""" 
@Time    : 2021/12/15 17:23
@Author  : Leo
@FileName: test.py
@SoftWare: PyCharm
@description: 
"""

import numpy as np

a = np.load('true.npy')
b = np.load('pred.npy')

print(a[0])
print("-----------------------------------------------")
print(b[0])