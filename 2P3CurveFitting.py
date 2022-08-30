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
name = "15-11_18_35, CONTROL 2 VK second tests"
fmt = ".csv"
svname = "Control 2"

#INPUT DISPLACEMENTS AND TRACKSPEED HERE
a = 0.15
b = 0.3
c = 0.45
d = 0.6
e = 0.75
f = 0.88
v = 10/60

#INPUT DEFAULT GUESS HERE, Th = guess1, Tc = guess2, k = guess3
guess1 = 51
guess2 = 24
guess3 = 0.01

#INPUT TOO HIGH OR TOO LOW BOUNDS FOR PREDICTION HERE
upbnd = 70
lbnd = 30
#%%
outputname = name + "results.csv"
avgoutname = name + "resultsavgd.csv"

raw = np.genfromtxt(name + fmt,dtype='float',delimiter=',',skip_header=2)
#print(raw)

temp1 = raw[:,0]
#print(temp1)
temp2 = raw[:,1]
#print(temp2)
temp3 = raw[:,2]
#print(temp3)
temp4 = raw[:,3]
temp5 = raw[:,4]
temp6 = raw[:,5]
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
#%%
mark = 84
A = temp1[mark]
#print(A)
B = temp2[mark]
#print(B)
C = temp3[mark]
#print(C)    
D = temp4[mark]
E = temp5[mark]
F = temp6[mark]

def fun(x, G, L, k):
    return  G * np.exp(-k*(x/v)) + L
xvals = np.array([a, b, c, d, e, f])
yvals = np.array([A, B, C, D, E, F])
try:
    opt, acc = curve_fit(fun, xvals, yvals, p0=guess_mast)
except Exception:
    print("couldn't find curve for point")
    pass

plt.scatter(xvals, yvals, label='Cooling Temperatures')
xrange = np.linspace(0, 1, 777)
try:
    plt.plot(xrange, fun(xrange, opt[0], opt[1], opt[2]),
             label='Fitted function')
except Exception:
    print("couldn't plot curve")
    pass
plt.legend(loc='best')
plt.show()
#%%

for j in vald:
    A = temp1[inx]
    #print(A)
    B = temp2[inx]
    #print(B)
    C = temp3[inx]
    #print(C)
    D = temp4[inx]
    E = temp5[inx]
    F = temp6[inx]
    xvals = np.array([a, b, c, d, e, f])
    yvals = np.array([A, B, C, D, E, F])
    
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
np.savetxt(svname, res, delimiter = ',')

#%%
#plotting prediction graph
datavals = len(vald)
sigfigdelta = "{:#.3g}".format(avgdelta)
time = np.linspace(0, 60, datavals)
plt.xlabel('time (seconds)') # creates a label 'x' for the x-axis
plt.ylabel('temperature (°C)') # creates a label 'y' for the y-axis
plt.plot(time, res[:,1], label="Validation")
plt.plot(time, res[:,0], label="Prediction")
plt.ylim(lbnd, upbnd)
title = 'Curve Fitting Algorithm (Exponential)' + ': ' + sigfigdelta
plt.title(title) # gives the plot a title
plt.legend()
plt.savefig(svname + "curvefitgraph.png", dpi=600)
plt.show()


        
