#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 17:44:51 2025

@author: lamsahel
"""

import numpy as np
import math

'''
This file contains the boundary and initial data to start the direct solver, 
as well as the bottom. Here, we propose several bottom examples, 
and one can control the inlet and outlet boundary conditions.
'''



#\\\\\\\\\\\\\\\\\\\\\\\\  Bottoms  \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
########################             ###############################"#################


'''
def bottom_f(x):     #classic bump 
    h_1=0.2
    w_1=2
    x_c=10
    if 8<=x<=12:
        return h_1-(h_1/w_1**2)*(x-x_c)**2
    else:
        return 0.
    

'''

'''
def bottom_f(x):    #sandbar  
     a =  0.1  #0.025
     b = 2
     c = 3*np.pi /5
     xr = (x / 25) * 6 - 3
     return a * (np.tanh(b * (xr+ c)) - np.tanh(b * (xr - c)))    
 

'''



def bottom_f(x): # Gaussian
     a = 0.2
     b=2.
     t = (x / 25) * 6 - 3
     return a / np.cosh(b * t)                      




'''
def bottom_f(x):        # complex bottom
    number_os=0.3
    
    if x<=np.pi/0.3:
       return (1+np.cos(number_os*x))*0.1
   
  #  elif 15 < x <= 20:
       
   #    return 0.1
   
    elif 15.0 <= x <= 19.0:
        h_2=0.3
        w_2=2.0
        c_centre2=17.0
        return h_2 - (h_2/(w_2**2))*(x-c_centre2)**2
    elif np.pi/0.3 <= x <= np.pi/0.3+4:
        h_2=0.3
        w_2=2.0
        c_centre2=(2*np.pi/0.3+4 )/2
        return h_2 - (h_2/(w_2**2))*(x-c_centre2)**2
   
    else:
       return 0.



'''


##### not in our  paper 
'''
def bottom_f(x):  # Flat bottom
    return 0.

'''

'''
def bottom_f(x):     #two bumps
    #bump between 8 and 12
    h_1=0.2
    w_1=2.
    c_centre1=10
    
    ##bump between 15 and 20
    h_2=0.3
    w_2=2.0
    c_centre2=15.0
    
    if 8 <= x <= 12:
        return h_1 - (h_1/(w_1**2))*(x-c_centre1)**2
    
    elif 13.0 <= x <= 17.0:
       return h_2 - (h_2/(w_2**2))*(x-c_centre2)**2
   
   # elif x>=17:
    #    return  h_2 - (h_2/(w_2**2))*(17-c_centre2)**2
    else:
         return 0
 
 '''   


'''
def bottom_f(x):       #sandbar by me
    H=0.3
    if x <= 0 or x >= 25:
        return 0.0
    elif x < 5:
        return 0.5*H*(1 - np.cos(np.pi * (x - 0)/5))
    elif x <= 20:
        return H
    else:  # 20 < x < 25
        return 0.5*H*(1 + np.cos(np.pi * (x - 20)/5))
'''  
'''
def bottom_f(x):    # for q_inlet =4.48 and + 0.02*t
    maxval=0.2
    
    if 5 <= x <= 15:
        return  (maxval/10)*(x-5)
        #return 0.2
    #elif 20<=x<=25:
        #return 0.2
    elif x>=15:
       return (maxval/10)*(10)
    else:
        return 0

'''



'''
def bottom_f(x): # cos bottom
    number_os=0.3
    return (1+np.cos(number_os*x))*0.1

'''




'''
#disc Bottom
def bottom_f(x):
    
    # (0, 2): "cos-like" quadratic with zeros at 1 and 2, value alpha1 at 0
    if  x <= 0 or x>=25:
        return  0


    elif 1 < x <= 10:
        
        return 0.15

    

    # (12.5, 16.5): parabolic bump centered at 14.5, half-width 2, height alpha5
    elif 12.5 < x <= 16.5:
        alpha5 = 0.2
        return alpha5 - (alpha5 / (2 ** 2)) * (x - 14.5) ** 2



    # Outside [0,25] and at the other intervales, return zero (or add the functions at the boundary)
    else:
        return 0.0


'''





#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  boundary conditions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#########################################                        ##################################
# Time‐varying inlet/outlet 

def q_inlet_Bc(t):   #sinusoidal
    Q_0=1.5
    A=0.2  #0.5
    period=10  #20
    return  Q_0+A * np.sin(2*np.pi * t / period)


    


'''
def q_inlet_Bc(t):   #inlet from https://www.researchgate.net/profile/Brett-Sanders/publication/245214119_Adjoint_Sensitivity_Analysis_for_Shallow-Water_Wave_Control/links/54ef63fd0cf25f74d7222bbf/Adjoint-Sensitivity-Analysis-for-Shallow-Water-Wave-Control.pdf
    Q_0=2.2627
    A=1.   #0.5
    x = 0.03 * (t - 120)     # at least take T>=200
    return Q_0 + A * (1.0 / np.cosh(x))**2      
    
'''

'''
def q_inlet_Bc(t):   # my sinusoidal
    Q_0=4.48
    A=0.4
    return  Q_0+A * np.sin(np.pi * t)

'''

########  boundary depth



def h_outlet_Bc(t):
    return 0.7


def h_inlet_Bc(t):               # needed for supercritical flow
    return 1



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ apply the boundary conditions function \\\\\\\\\\\\\\\\\\\\\\\_\
##################################                                       ########################

def boundary_cd(t,h,q):
    ###Upstream inlet x=0
    q[0]    =  q_inlet_Bc(t)#  q_inlet_Bc(t) #q[1]      # Dirichlet  
    h[0]    =h[1] #h[1]                  # Transmissive h_x(0)=0 or do‐nothing BC
        
    ##Downstream outlet          
    q[-1]   = q[-2]                # Transmissive q_x(L)=0
    h[-1]   =h[-2] #h_outlet_Bc(t)      # Dirichlet  or Transmissive
   
    ##if fully open channel replace q[0]=q[1] and h[-1]=h[-2], 
    ##(do‐nothing BC) for Upstream and Downstream for both  h and q in 0 and L
    
    #### the above BCs are for the case subcritical if we are at the Supercritical case we must do the following
           # superc at one boundary BC=0,L :
               #  Dirichlet on Bc for both q and h     #exp BC=0
               # Transmissive bc on BC^c for both a dn h   # the BC^c=L
          # Supercritical  in both 0 and L:
              # pure transmissive:  do‐nothing bc on both 0 and L
    



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Initial data \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
##########################################                ###################################

def initial_surface(x):
    h_0= h_outlet_Bc(0)
    A=0.15
    centre_s=5
    width_s=1.2
    return h_0 #+ A * np.exp(-((x - centre_s)/width_s)**2)
    N_waves = 5
    lambda_w = 25/ N_waves     # 12.5 m
    k = 2*np.pi / lambda_w

    #return h_0+ A * np.sin(k * x)
    
def inital_dat(Nx,Batxj): 
    L=25.
    X=np.linspace(0, L, Nx+1)
    eta_0=initial_surface(X)
    h = np.full(Nx+1, eta_0-Batxj)
    q = np.full(Nx+1, 1*q_inlet_Bc(0))
    
    return h,q







