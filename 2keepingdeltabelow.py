# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:40:04 2022

@author: Student2
"""

#%%
#imports
import numpy as np

import math as m
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%%
val1 = 50
val2 = 20

while val2-val1 > 10 or val2-val1<-10:
    if val2-val1 > 0:
        val2 -= 0.01
        print(val2)
    if val2-val1 < 0:
        val2 += 0.01
        print(val2)
print("final:", val2)