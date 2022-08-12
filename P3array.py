# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:56:00 2022

@author: Student2
"""
#%%
#imports
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.optimize import fsolve
import csv
import time

#%%
#INPUT DATA FILE NAME HERE
name = "ChangingHeatTest"
fmt = ".csv"

#INPUT DISPLACEMENTS AND TRACKSPEED HERE
a = 0.05
b = 0.1
c = 0.15
v = 3/60

#INPUT DEFAULT GUESS HERE, Th = guess1, Tc = guess2, k = guess3
guess1 = 50
guess2 = 25
guess3 = 0.01

#INPUT TOO HIGH OR TOO LOW BOUNDS FOR PREDICTION HERE
upbnd = 100
lbnd = 20
#%%
outputname = name + "results.csv"
avoutname = name + "resultsavgd.csv"
graphname = name + "graph.png"
graphnameav = name + "graphavgd.png"

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

#%%
for v in vald:
    #'''
    A = temp1[inx]
    #print(A)
    B = temp2[inx]
    #print(B)
    C = temp3[inx]
    #print(C)
    #'''
    #%%
    '''
    #Temp Averaging
    if inx == 0:
        A = temp1[inx]
        #print(A)
        B = temp2[inx]
        #print(B)
        C = temp3[inx]
        #print(C)
    elif inx == 1:
        inx1 = inx - 1
        A = np.mean([temp1[inx], temp1[inx1]])
        B = np.mean([temp2[inx], temp2[inx1]])
        C = np.mean([temp3[inx], temp3[inx1]])
    elif inx == 2:
        inx1 = inx - 1
        inx2 = inx - 2
        A = np.mean([temp1[inx], temp1[inx1], temp1[inx2]])
        B = np.mean([temp2[inx], temp2[inx1], temp2[inx2]])
        C = np.mean([temp3[inx], temp3[inx1], temp3[inx2]])
    elif inx == 3:
        inx1 = inx - 1
        inx2 = inx - 2
        inx3 = inx - 3
        A = np.mean([temp1[inx], temp1[inx1], temp1[inx2], temp1[inx3]])
        B = np.mean([temp2[inx], temp2[inx1], temp2[inx2], temp2[inx3]])
        C = np.mean([temp3[inx], temp3[inx1], temp3[inx2], temp3[inx3]])
    else:
        inx1 = inx - 1
        inx2 = inx - 2
        inx3 = inx - 3
        inx4 = inx - 4
        A = np.mean([temp1[inx], temp1[inx1], temp1[inx2], temp1[inx3], temp1[inx4]])
        B = np.mean([temp2[inx], temp2[inx1], temp2[inx2], temp2[inx3], temp2[inx4]])
        C = np.mean([temp3[inx], temp3[inx1], temp3[inx2], temp3[inx3], temp3[inx4]])
    '''
    #%%
    #Solving attempt 1, guess on from previous values
    try:
        def eqs(x):
            Th = x[0]
            Tc = x[1]
            k = x[2]
            
            f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
            g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
            h = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
            
            return [f, g, h]
        guess = solution
        solution = fsolve(eqs, guess)
        print("trial values solution [T_h, T_c, k]: ", solution)
    except Exception:
        print("error: numerical solving 1")
        pass
#%%    
    #attempt 2, guess from previous for Th, but fixed guesses for Tc & k
    try:
        def eqs(x):
            Th = x[0]
            Tc = x[1]
            k = x[2]
            f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
            g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
            h = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
            
            return [f, g, h]
        guess = np.array([solution[0], guess2, guess3])
        solution = fsolve(eqs, guess)
        print("trial values retry1 solution [T_h, T_c, k]: ", solution)
    except Exception:
        print("error: numerical solving 2")
        pass    
    
#%%
    #'''
    #solving attempt 3, if value still to high or too low, try with fixed guess values
    if solution[0]<lbnd or solution[0]>upbnd:
        try:
            def eqs(x):
                Th = x[0]
                Tc = x[1]
                k = x[2]
                f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
                g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
                h = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
                
                return [f, g, h]
            guess = np.array([guess1, guess2, guess3])
            solution = fsolve(eqs, guess)
            print("trial values retry2 solution [T_h, T_c, k]: ", solution)
        except Exception:
            print("error: numerical solving 3")
            pass    
    else:
        pass
    #'''

#%%       
    #appending array
    pred = solution[0]
    val = vald[inx]
    delta = pred - val
    resline = np.array([pred, val, delta])
    print(resline)
    #print(np.shape(resline))
    res = np.append(res, [resline], axis=0)
    
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
plt.savefig(graphname, dpi=600)

#%%
#Predictions averaged out of the previous 2 values and current one
Ths = res[:,0]
Ths_smooth = np.empty(0)
deltas_smooth = np.empty(0)
index = 0
for x in Ths:
    index1 = index - 1
    index2 = index - 2
    index3 = index - 3 
    index4 = index - 4 
    if index == 0:
        Thavg = Ths[index]
        Ths_smooth = np.append(Ths_smooth, Thavg)
        index += 1
    elif index == 1:
        Thavg = np.mean([Ths[index], Ths[index1]])
        Ths_smooth = np.append(Ths_smooth, Thavg)
        index += 1
    elif index == 2:
        Thavg = np.mean([Ths[index], Ths[index1], Ths[index2]])
        Ths_smooth = np.append(Ths_smooth, Thavg)
        index += 1
    elif index == 3:
        Thavg = np.mean([Ths[index], Ths[index1], Ths[index2], Ths[index3]])
        Ths_smooth = np.append(Ths_smooth, Thavg)
        index += 1
    else:
        Thavg = np.mean([Ths[index], Ths[index1], Ths[index2], Ths[index3], Ths[index4]])
        Ths_smooth = np.append(Ths_smooth, Thavg)
        index += 1        
#print(Ths)
#print(Ths_smooth)
res[:,0] = Ths_smooth
#%%
#updating deltas
indexx = 0
for x in Ths:
    delta_smooth = res[indexx, 0] - res[indexx, 1]
    deltas_smooth = np.append(deltas_smooth, delta_smooth)
    indexx += 1
res[:,2] = deltas_smooth
print("averaged results:", res)
#%%
#calculating new average delta
adeltas_smooth = abs(deltas_smooth)
#print(adeltas)
#fin = len(adeltas)+1
avgdelta_smooth = np.mean(adeltas_smooth)
print(avgdelta_smooth)
np.savetxt(avoutname, res, delimiter = ',')

#%%
#plotting averaged graph
plt.plot(time, res[:,1], label="Validation")
plt.plot(time, res[:,0], label="Prediction")
plt.ylim(lbnd, upbnd)
plt.savefig(graphnameav, dpi=600)
plt.legend()
plt.show()

        
