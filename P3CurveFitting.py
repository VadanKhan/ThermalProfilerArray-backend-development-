# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:18:43 2022

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
#INPUT DATA FILE NAME HERE
name = "ControlTest"
fmt = ".csv"

#INPUT DISPLACEMENTS AND TRACKSPEED HERE
a = 0.3
b = 0.6
c = 0.88
v = 5/60

#INPUT DEFAULT GUESS HERE, Th = guess1, Tc = guess2, k = guess3
guess1 = 51
guess2 = 24
guess3 = 0.01

#INPUT TOO HIGH OR TOO LOW BOUNDS FOR PREDICTION HERE
upbnd = 100
lbnd = 20
#%%
outputname = name + "results.csv"
avgoutname = name + "resultsavgd.csv"

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

res = np.empty((0,3))
#print(res)

solution = np.array([guess1, guess2, guess3])

guess_mast = np.array([guess1-guess2, guess2, guess3])
#print("guess =", guess)

#Th2 = fun(0, opt[0], opt[1], opt[2])
#print("Th2: ", Th2)
'''
plt.scatter(xvals, yvals, label='Data')
xrange = np.linspace(1, 4, 777)
plt.plot(xrange, fun(xrange, opt[0], opt[1], opt[2]),
         label='Fitted function')
plt.legend(loc='best')
plt.show()
'''
#%%

for j in vald:
    A = temp1[inx]
    #print(A)
    B = temp2[inx]
    #print(B)
    C = temp3[inx]
    #print(C)    
    def fun(x, G, L, k):
        return  G * np.exp(-k*(x/v)) + L
    xvals = np.array([a, b, c])
    yvals = np.array([A, B, C])
    
    #rolling
    try:
        guess = solution
        opt, acc = curve_fit(fun, xvals, yvals, p0=guess)
        #print("G, L, k:", opt)
        
        Tc = opt[1]
        Th = opt[0] + opt[1]
        k = opt[2]
        print("Th, Tc, k:", Th, " ", Tc, " ", k)
        solution = np.array([Th, Tc, k])
        
    except Exception:
        print("error: numerical solving 1")
        pass
    
    #rolling with fixed Tc and Th
    try:
        guess = np.array([solution[0], guess_mast[1], guess_mast[2]])
        opt, acc = curve_fit(fun, xvals, yvals, p0=guess)
        #print("G, L, k:", opt)
        
        Tc = opt[1]
        Th = opt[0] + opt[1]
        k = opt[2]
        print("Th, Tc, k:", Th, " ", Tc, " ", k)
        solution = np.array([Th, Tc, k])
        print("trial values retry1 solution [T_h, T_c, k]: ", solution)
    except Exception:
        print("error: numerical solving 2")
        pass
    
    #fixed guess
    try:
        guess = guess_mast
        opt, acc = curve_fit(fun, xvals, yvals, p0=guess)
        #print("G, L, k:", opt)
        
        Tc = opt[1]
        Th = opt[0] + opt[1]
        k = opt[2]
        print("Th, Tc, k:", Th, " ", Tc, " ", k)
        solution = np.array([Th, Tc, k])
        
        print("trial values retry1 solution [T_h, T_c, k]: ", solution)
    except Exception:
        print("error: numerical solving 2")
        pass
#%%       
    #appending array
    try:
        pred = solution[0]
        val = vald[inx]
        delta = pred - val
        resline = np.array([pred, val, delta])
        print(resline)
        #print(np.shape(resline))
        res = np.append(res, [resline], axis=0)
    except Exception:
        pass
    
    
    inx += 1
    print(inx)

#%%

print(res)
#print(res[119,0])
deltas = res[:,2]
#print(deltas)
adeltas = abs(deltas)
#print(adeltas)
#fin = len(adeltas)+1
avgdelta = np.mean(adeltas)
print(avgdelta)
np.savetxt(outputname, res, delimiter = ',')

#%%
#plotting prediction graph
time = np.linspace(0, 60, 120)
plt.plot(time, res[:,1], label="Validation")
plt.plot(time, res[:,0], label="Prediction")
plt.ylim(lbnd, upbnd)
plt.legend()
plt.show()
plt.savefig(name + "curvefitgraph.png", dpi=600)

        
