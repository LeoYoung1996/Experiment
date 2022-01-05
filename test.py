"""
@Time    : 2021/12/23 6:47 下午
@Author  : LeoYoung
@FileName: test.py
@Software: PyCharm
@description: 任何程序的测试
"""

import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
print(torch.__version__)
print(torch.cuda.is_available())