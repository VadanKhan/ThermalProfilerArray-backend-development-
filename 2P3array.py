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
from scipy.optimize import curve_fit
import csv
import time
import sys

#%% INPUTS
#INPUT DATA FILE NAME HERE
name = "Control 1 random"
fmt = ".csv"
#INPUT NAME OF SAVED PNG YOU WANT
svname = "Control 1 random"
svnameav = svname + " avg"

#INPUT DISPLACEMENTS AND TRACKSPEED HERE
a = 0.15
b = 0.3
c = 0.45
d = 0.6
e = 0.75
f = 0.88
v = 10/60

#INPUT DEFAULT GUESS HERE, Th = guess1, Tc = guess2, k = guess3
guess1 = 50
guess2 = 25
guess3 = 0.01

#INPUT DEFAULT LINEAR GUESS HERE:
lingrad = -5
linintcpt = 50

#INPUT TOO HIGH OR TOO LOW BOUNDS FOR PREDICTION HERE
upbnd = 70
lbnd = 30

#AND MAXMIMUM TOLERATED JUMP IN TEMPERATURE
maxjump = 10

#ADD WEIGHT AS DECIMAL FRACTION OF SOLUTION DETERMINED BY LINEAR FIT (initial)
setting = 0.5

#FRACTION OF NET DELTA TO ABS DELTA BEFORE MACHING TRIES LEARNING
netabsfrac = 0.333

#INITIAL DIRECTION FOR SELF IMPROVEMENT ALGORITHM, True for try to decrease linear input and vice versa
declin = True

#TEMP THAT HEATER MUST BE SWITCHED BACK ON, & TARGET TEMP
heat = 40
#%% Startup
outputname = svname + "results.csv"
avoutname = svname + "resultsavgd.csv"
graphname = svname + "graph.png"
graphnameav = svname + "graphavgd.png"

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

#jump logs
chnginx = np.empty(0)
settinglog = np.empty(0)
jumplog = np.empty(0)

#swtich on off logs
heaterswitch = np.empty(0)

res = np.empty((0,3))
#print(res)

solution = np.array([guess1, guess2, guess3])
guess_mast = np.array([guess1, guess2, guess3])
linguess_mast = np.array([lingrad, linintcpt ])
setting_mast = setting

#%% Begin Loop
for v in vald:
    #'''
    A = temp1[inx]
    #print(A)
    B = temp2[inx]
    #print(B)
    C = temp3[inx]
    #print(C)
    D = temp4[inx]
    E = temp5[inx]
    F = temp6[inx]
    #'''
    #%% Input Temp Averaging
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
#%% COMBINATION 1
    def eqs(x):
        Th = x[0]
        Tc = x[1]
        k = x[2]
        
        f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
        g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
        h = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
        
        return [f, g, h]
#%% Solving attempt 1, guess on from previous values
    try:
        guess = solution
        solution = fsolve(eqs, guess)
        print("trial values solution [T_h, T_c, k]: ", solution)
        fail = True
    except Exception:
        print("error: numerical solving 1")
        fail = True
        
#%% Run 2, guess from previous for Th, but fixed guesses for Tc & k
    if fail == True:
        try:
            guess = np.array([solution[0], guess2, guess3])
            solution = fsolve(eqs, guess)
            print("trial values retry1 solution [T_h, T_c, k]: ", solution)
            fail = False
        except Exception:
            print("error: numerical solving 2")
            fail = True
            pass

#%% solving attempt 3, try with fixed guess values
    if fail == True:
        try:
            guess = np.array([guess1, guess2, guess3])
            solution = fsolve(eqs, guess)
            print("trial values retry2 solution [T_h, T_c, k]: ", solution)
        except Exception:
            print("error: numerical solving 3")
            pass    
    else:
        pass
    #'''

#%% reducing jumps in temperature to a certain max jump
    inx1 = inx - 1
    if inx > 0:
        jump = solution[0] - res[inx1,0]
        if abs(jump)>maxjump and abs(jump)<1000 and solution[0]<100 and solution[0]>20:
            jumpreduce = True
        else: jumpreduce = False
        while abs(jump)>maxjump and abs(jump)<1000 and solution[0]<100 and solution[0]>20:
            if jump>0:
                solution[0] -= 1
                jump = solution[0] - res[inx1,0]
                print("jump reduce: ", jump)
            if jump<0:
                solution[0] += 1
                jump = solution[0] - res[inx1,0]
                print("jump reduce: ", jump)
        if jumpreduce == True:
            print("jump reduced Th: ", solution[0])
    
#%% resetting anomalous values
    inx1 = inx - 1   
    '''
    if solution[2]>10 or solution[2]<-10:
        solution[2] = guess3
        print("RESET K")

    if solution[1]>100 or solution[1]<0:
        solution[1] = guess2
        print("RESET Tc")
    '''
    
    if solution[0]>100 or solution[0]<20:
        solution[0] = res[inx1,0]
        print("RESET Th")

#%% COMBINATION 2
    def eqs(x):
        Th = x[0]
        Tc = x[1]
        k = x[2]
        
        f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
        g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
        h = Tc + (Th - Tc)*m.exp(-k*(d/v)) - D
        
        return [f, g, h]

#%% 2: Solving attempt 1, guess on from previous values
    try:
        guess = solution
        solution2 = fsolve(eqs, guess)
        print("trial values solution2 [T_h, T_c, k]: ", solution2)
        fail = True
    except Exception:
        print("error: numerical solving 1")
        fail = True
        
#%% 2: guess from previous for Th, but fixed guesses for Tc & k
    if fail == True:
        try:
            guess = np.array([solution[0], guess2, guess3])
            solution2 = fsolve(eqs, guess)
            print("trial values retry1 solution2 [T_h, T_c, k]: ", solution2)
            fail = False
        except Exception:
            print("error: numerical solving 2")
            fail = True
            pass

#%% 2: solving attempt 3, try with fixed guess values
    if fail == True:
        try:
            guess = np.array([guess1, guess2, guess3])
            solution2 = fsolve(eqs, guess)
            print("trial values retry2 solution2 [T_h, T_c, k]: ", solution2)
        except Exception:
            print("error: numerical solving 3")
            pass    
    else:
        pass
    #'''

#%% 2: reducing jumps in temperature to a certain max jump
    inx1 = inx - 1
    if inx > 0:
        jump = solution2[0] - res[inx1,0]
        if abs(jump)>maxjump and abs(jump)<1000 and solution2[0]<100 and solution2[0]>20:
            jumpreduce = True
        else: jumpreduce = False
        while abs(jump)>maxjump and abs(jump)<1000 and solution2[0]<100 and solution2[0]>20:
            if jump>0:
                solution2[0] -= 1
                jump = solution2[0] - res[inx1,0]
                print("jump reduce: ", jump)
            if jump<0:
                solution2[0] += 1
                jump = solution2[0] - res[inx1,0]
                print("jump reduce: ", jump)
        if jumpreduce == True:
            print("jump reduced Th: ", solution2[0])
    
#%% 2: resetting anomalous values
    inx1 = inx - 1   
    '''
    if solution[2]>10 or solution[2]<-10:
        solution[2] = guess3
        print("RESET K")

    if solution[1]>100 or solution[1]<0:
        solution[1] = guess2
        print("RESET Tc")
    '''
    
    if solution2[0]>100 or solution2[0]<20:
        solution2 = np.empty(0)

#%% COMBINATION 3
    def eqs(x):
        Th = x[0]
        Tc = x[1]
        k = x[2]
        
        f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
        g = Tc + (Th - Tc)*m.exp(-k*(b/v)) - B
        h = Tc + (Th - Tc)*m.exp(-k*(e/v)) - E
        
        return [f, g, h]
#%% 3: Solving attempt, guess on from previous values
    try:
        guess = solution
        solution3 = fsolve(eqs, guess)
        print("trial values solution3 [T_h, T_c, k]: ", solution3)
        fail = True
    except Exception:
        print("error: numerical solving3 1")
        fail = True
        
#%% 3: guess from previous for Th, but fixed guesses for Tc & k
    if fail == True:
        try:
            guess = np.array([solution[0], guess2, guess3])
            solution3 = fsolve(eqs, guess)
            print("trial values retry1 solution3 [T_h, T_c, k]: ", solution3)
            fail = False
        except Exception:
            print("error: numerical solving3 2")
            fail = True
            pass

#%% 3: solving attempt 3, try with fixed guess values
    if fail == True:
        try:
            guess = np.array([guess1, guess2, guess3])
            solution3 = fsolve(eqs, guess)
            print("trial values retry2 solution3 [T_h, T_c, k]: ", solution3)
        except Exception:
            print("error: numerical solving3 3")
            pass    
    else:
        pass
    #'''

#%% 3: reducing jumps in temperature to a certain max jump
    inx1 = inx - 1
    if inx > 0:
        jump = solution3[0] - res[inx1,0]
        if abs(jump)>maxjump and abs(jump)<1000 and solution3[0]<100 and solution3[0]>20:
            jumpreduce = True
        else: jumpreduce = False
        while abs(jump)>maxjump and abs(jump)<1000 and solution3[0]<100 and solution3[0]>20:
            if jump>0:
                solution3[0] -= 1
                jump = solution3[0] - res[inx1,0]
                print("jump reduce: ", jump)
            if jump<0:
                solution3[0] += 1
                jump = solution3[0] - res[inx1,0]
                print("jump reduce: ", jump)
        if jumpreduce == True:
            print("jump reduced Th: ", solution3[0])
    
#%% 3: resetting anomalous values
    inx1 = inx - 1   
    '''
    if solution[2]>10 or solution[2]<-10:
        solution[2] = guess3
        print("RESET K")

    if solution[1]>100 or solution[1]<0:
        solution[1] = guess2
        print("RESET Tc")
    '''
    
    if solution3[0]>100 or solution3[0]<20:
        solution3 = np.empty(0)
#%% COMBINATION 4
    def eqs(x):
        Th = x[0]
        Tc = x[1]
        k = x[2]
        
        f = Tc + (Th - Tc)*m.exp(-k*(a/v)) - A
        g = Tc + (Th - Tc)*m.exp(-k*(c/v)) - C
        h = Tc + (Th - Tc)*m.exp(-k*(f/v)) - F
        
        return [f, g, h]
#%% 4: Solving attempt 1, guess on from previous values
    try:
        guess = solution
        solution4 = fsolve(eqs, guess)
        print("trial values solution4 [T_h, T_c, k]: ", solution4)
        fail = True
    except Exception:
        print("error: numerical solving4 1")
        fail = True
        
#%% 4: guess from previous for Th, but fixed guesses for Tc & k
    if fail == True:
        try:
            guess = np.array([solution[0], guess2, guess3])
            solution4 = fsolve(eqs, guess)
            print("trial values retry1 solution4 [T_h, T_c, k]: ", solution4)
            fail = False
        except Exception:
            print("error: numerical solving4 2")
            fail = True
            pass

#%% 4: solving attempt 3, try with fixed guess values
    if fail == True:
        try:
            guess = np.array([guess1, guess2, guess3])
            solution4 = fsolve(eqs, guess)
            print("trial values retry2 solution4 [T_h, T_c, k]: ", solution4)
        except Exception:
            print("error: numerical solving4 3")
            solution4 = np.empty(0)
            pass    
    else:
        pass
    #'''

#%% 4: reducing jumps in temperature to a certain max jump
    inx1 = inx - 1
    try:
        if inx > 0:
            jump = solution4[0] - res[inx1,0]
            if abs(jump)>maxjump and abs(jump)<1000 and solution4[0]<100 and solution4[0]>20:
                jumpreduce = True
            else: jumpreduce = False
            while abs(jump)>maxjump and abs(jump)<1000 and solution4[0]<100 and solution4[0]>20:
                if jump>0:
                    solution4[0] -= 1
                    jump = solution4[0] - res[inx1,0]
                    print("jump reduce: ", jump)
                if jump<0:
                    solution4[0] += 1
                    jump = solution4[0] - res[inx1,0]
                    print("jump reduce: ", jump)
            if jumpreduce == True:
                print("jump reduced Th: ", solution4[0])
    except Exception:
        print("Error Reducing Jump")
        pass
    
#%% 4: resetting anomalous values
    inx1 = inx - 1   
    '''
    if solution[2]>10 or solution[2]<-10:
        solution[2] = guess3
        print("RESET K")

    if solution[1]>100 or solution[1]<0:
        solution[1] = guess2
        print("RESET Tc")
    '''
    try:
        if solution4[0]>100 or solution4[0]<20:
            solution4[0] = np.empty(0)
            print("RESET Th", solution4[0])
    except Exception:
        print("couldn't reset solution4")
        pass
#%% linear fit solving
    def fun(x, M, K):
        return  M * (x/v) + K
    xvals = np.array([a, b, c, d, e, f])
    yvals = np.array([A, B, C, D, E, F])
    try:
        linguess = linguess_mast
        opt, acc = curve_fit(fun, xvals, yvals, p0=linguess)
        #print("G, L, k:", opt)
        
        Th = opt[1]
        Tc = 25
        k = 0.01
        print("Th, Tc, k:", Th, " ", Tc, " ", k)
        solutionlin = np.array([Th, Tc, k])
        
        print("trial values linear fit solution [T_h, T_c, k]: ", solutionlin)
    except Exception:
        print("error: linear solving")
        pass
#%% Linear Fit Reduce Jump
    inx1 = inx - 1
    try:
        if inx > 0:
            jump = solutionlin[0] - res[inx1,0]
            if abs(jump)>maxjump and abs(jump)<1000 and solutionlin[0]<100 and solutionlin[0]>20:
                jumpreduce = True
            else: jumpreduce = False
            while abs(jump)>maxjump and abs(jump)<1000 and solutionlin[0]<100 and solutionlin[0]>20:
                if jump>0:
                    solutionlin[0] -= 1
                    jump = solutionlin[0] - res[inx1,0]
                    print("jump reduce: ", jump)
                if jump<0:
                    solutionlin[0] += 1
                    jump = solutionlin[0] - res[inx1,0]
                    print("jump reduce: ", jump)
            if jumpreduce == True:
                print("jump reduced Th: ", solutionlin[0])
    except Exception:
        print("Error Reducing Jump")
        pass
#%% Linear fit anomalous reset
    inx1 = inx - 1   
    '''
    if solution[2]>10 or solution[2]<-10:
        solution[2] = guess3
        print("RESET K")

    if solution[1]>100 or solution[1]<0:
        solution[1] = guess2
        print("RESET Tc")
    '''
    try:
        if solutionlin[0]>100 or solutionlin[0]<20:
            solutionlin[0] = res[inx1,0]
            print("RESET Th", solution4[0])
    except Exception:
        print("couldn't reset solutionlin")
        pass
#%% Weighting Solution Line
    #print(type(solution[0]))
    #print(solution[0])
    #print(type(solution2[0]))
    #print(solution2[0])
    solutionset = np.array([solution[0]])
    try:
        solutionset = np.append(solutionset, solution2[0])
    except Exception:
        print("could not add solution2")
        pass
    try:
        solutionset = np.append(solutionset, solution3[0])
    except Exception:
        print("could not add solution3")
        pass
    try:
        solutionset = np.append(solutionset, solution4[0])
    except Exception:
        print("could not add solution4")
        pass
    nums = np.mean(solutionset)
    try:
        solutions = np.array([nums, solutionlin[0]])
    except Exception:
        print("could not add solutionlin")
        pass
    weightlin = setting
    weightnum = 1 - weightlin
    pred = weightlin * solutionlin[0] + weightnum * nums
#%% appending array
    val = vald[inx]
    delta = pred - val
    resline = np.array([pred, val, delta])
    print(resline)
    #print(np.shape(resline))
    res = np.append(res, [resline], axis=0)
#%% Evaluation of last 5 jumps and Reinforcement Machine Learning Iteration
    
    def isMultiple(num,  check_with):
        return num % check_with == 0;
    if (isMultiple(inx, 5)==True) and (inx>0):
        inx1 = inx - 1
        inx2 = inx - 2
        inx3 = inx - 3
        inx4 = inx - 4
        inx5 = inx - 5
        jump1 = pred - res[inx1,0]
        jump2 = res[inx1,0] - res[inx2,0]
        jump3 = res[inx2,0] - res[inx3,0]
        jump4 = res[inx3,0] - res[inx4,0]
        jump5 = res[inx4,0] - res[inx5,0]
        jumps = np.array([jump1, jump2, jump3, jump4, jump5])
        absjumps = abs(jumps)
        sumjumps = np.sum(jumps)
        sumabsjumps = np.sum(absjumps)
        jumplog = np.append(jumplog, sumabsjumps)
        highthresh = 0.5
        '''
        if sumjumps > (highthresh*len(jumps)*maxjump):
            setting -= 0.1
            chnginx = np.append(chnginx, inx)
        '''
        #code that says if the jumps have increased, flip the direction we are iterating
        
        if len(jumplog)>2:
            jumpdiff1 = jumplog[len(jumplog)-1] - jumplog[len(jumplog)-2]
            jumpdiff2 = jumplog[len(jumplog)-2] - jumplog[len(jumplog)-3]
            if jumpdiff1>0 and jumpdiff2>0:
                declin ^= True
        
        if sumjumps < netabsfrac * sumabsjumps and setting >= 0.2 and declin == True:
            setting -= 0.1
            settinglog = np.append(settinglog, setting)
            chnginx = np.append(chnginx, inx)
        if sumjumps < netabsfrac * sumabsjumps and setting <= 0.8 and declin == False:
            setting += 0.1
            settinglog = np.append(settinglog, setting)
            chnginx = np.append(chnginx, inx)
        
#%% response to temperature
    if pred < heat:
        heaterswitch = np.append(heaterswitch, 1)
    if pred > heat:
        heaterswitch = np.append(heaterswitch, 0)
    
    inx += 1
    print(inx)

#%% output
print(res)
#print(res[119,0])
deltas = res[:,2]
#print(deltas)
adeltas = abs(deltas)
#print(adeltas)
#fin = len(adeltas)+1
avgdelta = np.mean(adeltas)
#print(avgdelta)
np.savetxt(outputname, res, delimiter = ',')

#%% plotting prediction graph
datapnts = len(vald)
sigfigdelta = "{:#.3g}".format(avgdelta)
print(sigfigdelta)
time = np.linspace(0, 60, datapnts)
plt.plot(time, vald, label="Validation")
plt.plot(time, res[:,0], label="Prediction")
plt.xlabel('time (seconds)') # creates a label 'x' for the x-axis
plt.ylabel('temperature (°C)') # creates a label 'y' for the y-axis
plt.ylim(lbnd, upbnd)
title = 'Array: 6 sensor Results, ' + svname + ': ' + sigfigdelta
plt.title(title)
plt.legend()
plt.savefig(graphname, dpi=777)
plt.show()

#%% Predictions averaged out of the previous 2 values and current one
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
#%% updating deltas
indexx = 0
for x in Ths:
    delta_smooth = res[indexx, 0] - res[indexx, 1]
    deltas_smooth = np.append(deltas_smooth, delta_smooth)
    indexx += 1
res[:,2] = deltas_smooth
print("averaged results:", res)
#%% calculating new average delta
adeltas_smooth = abs(deltas_smooth)
#print(adeltas)
#fin = len(adeltas)+1
avgdelta_smooth = np.mean(adeltas_smooth)
print(avgdelta)
print(avgdelta_smooth)
np.savetxt(avoutname, res, delimiter = ',')

#%% plotting averaged graph
datapnts = len(vald)
sigfigdeltaav = "{:#.3g}".format(avgdelta_smooth)
print(sigfigdelta)
time = np.linspace(0, 60, datapnts)
plt.plot(time, vald, label="Validation")
plt.plot(time, Ths_smooth, label="Prediction")
plt.xlabel('time (seconds)') # creates a label 'x' for the x-axis
plt.ylabel('temperature (°C)') # creates a label 'y' for the y-axis
plt.ylim(lbnd, upbnd)
title = 'Array: 3 sensor Numerical Results, ' + svnameav + ': ' + sigfigdeltaav
plt.title(title)
plt.legend()
plt.savefig(graphnameav, dpi=777)
plt.show()

print("heating switch: ", heaterswitch)
print("final setting = ", setting)
print("change points: ", chnginx)
print("jump log: ", jumplog)
print("settings changes: ", settinglog)
#print(datapnts)
#print(len(heaterswitch))

        
