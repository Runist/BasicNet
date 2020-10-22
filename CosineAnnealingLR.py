# -*- coding: utf-8 -*-
# @File : CosineAnnealingLR.py
# @Author: Runist
# @Time : 2020-10-10 16:18
# @Software: PyCharm
# @Brief:

import numpy as np
from matplotlib import pyplot as plt


Lmin = 1e-8
Lmax = 1e-4


epoch = 50
learning_rate = []
warmth_period = 1

for e in range(1, epoch + 1):
    if e <= 4:
        lr = Lmax / 4 * e
    else:
        lr = Lmin + 0.5 * (Lmax - Lmin) * (1 + np.cos(warmth_period * e / epoch * np.pi))
    learning_rate.append(lr)
    print(e, lr)

plt.plot(range(epoch), learning_rate)
plt.show()
