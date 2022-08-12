# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:04:49 2022

@author: Student2
"""

#import serial
import numpy as np
import matplotlib as plt
import math as m
from scipy.optimize import fsolve
import csv
import time

#INPUT DATA FILE NAME HERE
name = "spacing50"
fmt = ".csv"

outputname = name + "results.csv"

raw = np.genfromtxt(name + fmt,dtype='float',delimiter=',',skip_header=0)
#print(raw)

temp1 = raw[:,0]
#print(temp1)
temp2 = raw[:,1]
#print(temp2)
temp3 = raw[:,2]
#print(temp3)
vald = raw[:,6]
#print(vald)

inx = 0


a = 0.05
b = 0.1
c = 0.15
v = 3/60

res = np.array([[0, 0, 0],
               [0, 0, 0]])
#print(res)

guess1 = 0.01
guess2 = 50
guess3 = 25


solution = np.array([guess1, guess2, guess3])
#%%
for v in vald:
    #solving
    try:
        A = temp1[inx]
        print(A)
        B = temp2[inx]
        print(B)
        C = temp3[inx]
        print(C)
        
        def eqs(x):
            k = x[0]
            Th = x[1]
            Tc = x[2]
            
            f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
            g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
            h = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
            
            return [f, g, h]
        guess = solution
        solution = fsolve(eqs, guess)
        print("trial values solution [k, T_h, T_c]: ", solution)
    except Exception:
        print("error: numerical solving")
        pass
#%%    
    #'''
    try:
        A = temp1[inx]
        print(A)
        B = temp2[inx]
        print(B)
        C = temp3[inx]
        print(C)
        
        def eqs(x):
            k = x[0]
            Th = x[1]
            Tc = x[2]
            f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
            g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
            h = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
            
            return [f, g, h]
        guess = np.array([guess1, solution[1], guess3])
        solution = fsolve(eqs, guess)
        print("trial values retry solution [k, T_h, T_c]: ", solution)
    except Exception:
        print("error: numerical solving")
        pass    
    #'''
#%%       
    #appending array
    pred = solution[1]
    val = vald[inx]
    delta = pred - val
    resline = np.array([pred, val, delta])
    print(resline)
    print(np.shape(resline))
    res = np.append(res, [resline], axis=0)    
    
    inx += 1
    print(inx)

#%%
print(res)
deltas = res[:,2]
#print(deltas)
adeltas = abs(deltas)
#print(adeltas)
fin = len(adeltas)+1
avgdelta = np.mean(adeltas[2:fin])
print(avgdelta)
np.savetxt(outputname, res, delimiter = ',')
