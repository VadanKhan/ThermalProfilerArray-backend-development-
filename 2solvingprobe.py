# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:54:04 2022

@author: Student2
"""
import numpy as np
import matplotlib as plt
import math as m
from scipy.optimize import fsolve
import csv
import time

a = 1
b = 2
c = 3
d = 4
v = 1

A = 25 + (50-25)*m.exp(-1) + 0.01
#print(A)
B = 25 + (50-25)*m.exp(-2) - 0.07
#print(B)
C = 25 + (50-25)*m.exp(-3) + 0.05
#print(C)
D = 25 + (50-25)*m.exp(-4) + 0.1

def eqs4(x):
    k = x[0]
    Tc = x[1]
    Th = x[2]
    f = np.empty(3)
    f[0] = Tc + (Th - Tc)*m.exp(-k*(a/v)) - float(A)
    f[1] = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
    f[2] = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
    #f[3] = Tc + (Th - Tc)*m.exp(-k*(d/v)) - D
    Tc > 0
    Th > 20
    Th < 150
    return f
guess = np.array([0.1, 20, 40])
solution = fsolve(eqs4, guess)
print("trial values solution [k, T_c, T_h]: ", solution)

def eqs(x):
    k = x[0]
    Tc = x[1]
    Th = x[2]
    f = np.empty(3)
    #f[0] = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
    f[0] = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
    f[1] = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
    f[2] = Tc + (Th - Tc)*m.exp(-k*(d/v)) - D
    Tc > 0
    Th > 20
    Th < 150
    return f
guess = np.array([0.1, 20, 40])
solution1 = fsolve(eqs, guess)
print("trial values solution [k, T_c, T_h]: ", solution1)

def eqs2(x):
    k = x[0]
    Tc = x[1]
    Th = x[2]
    f = np.empty(3)
    f[0] = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
    #f[0] = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
    f[1] = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
    f[2] = Tc + (Th - Tc)*m.exp(-k*(d/v)) - D
    Tc > 0
    Th > 20
    Th < 150
    return f
guess = np.array([0.1, 20, 40])
solution2 = fsolve(eqs2, guess)
print("trial values solution [k, T_c, T_h]: ", solution2)

def eqs3(x):
    k = x[0]
    Tc = x[1]
    Th = x[2]
    f = np.empty(3)
    f[0] = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
    f[1] = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
    #f[1] = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
    f[2] = Tc + (Th - Tc)*m.exp(-k*(d/v)) - D
    Tc > 0
    Th > 20
    Th < 150
    return f
guess = np.array([0.1, 20, 40])
solution3 = fsolve(eqs3, guess)
print("trial values solution [k, T_c, T_h]: ", solution3)

avgTh = np.mean([solution[2], solution1[2], solution2[2], solution3[2]])
print(avgTh)
