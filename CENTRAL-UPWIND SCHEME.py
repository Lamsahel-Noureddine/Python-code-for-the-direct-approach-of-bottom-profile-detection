#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:48:40 2025

@author: lamsahel
"""

 #\\\\\\\\\\\\\\\subject of the code\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\|\\\\\\\\
"""
In this code, we discretize the 1-d shallow water system using a second-order,
well-balanced, positivity-preserving central-upwind scheme,
following the algorithm in :
    
    https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=A+SECOND-ORDER+WELL-BALANCED+POSITIVITY+PRESERVING+CENTRAL-UPWIND+SCHEME+FOR+THE+SAINT-VENANT+SYSTEM&btnG=

Once we solve the direct solver, we use the surface data and the inlet discharge to
construct a direct approch solver for the inverse problem of bottom detection. 
The inverse solver requires knowledge of the surface profile and the inlet discharge 
over a small time interval, which allows us to compute $q$ and $q_t$ at a specific instant $t_0$,
such that, at $t_0$, we have $q(x,t_0) > \beta > 0$. This condition that can be satisfied either 
 by the proposition in our  work or numerically. 
 Moreover:
In our numerical results, we observe that even with a small positive inlet discharge
 taking enough time leads to strictly positive discharge throughout the entire domain, 
 even for complex bottom geometries.

"""
#\\\\\\\\\\\\\\\\\\ needed packages \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
import numpy as np
import matplotlib.pyplot as plt
from neededfunctionsforcode import *
from Bottom_inital_BCdatas import *
from Plotsfunctions import *
from invirseProblem_Solvers import *
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import pandas as pd

#\\\\\\\\\\\\\\\\\\\\\ Parameters \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
g     =9.812       # gravity
L     =25.         # domain length [a_1,a_2]
Nx    =100        # number of interior cells  
# when Nx large , we get good app results becouse we are more close to the exact free surface solution
dx    = L / Nx
x     = np.linspace(dx/2, L-dx/2, Nx)  # cell centers
X     = np.linspace(0, L, Nx+1)  # physical points
T     =20  # time  s   noted in paper t_f







#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\   Direct Solver      \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#################################                                ########################


'''
Recall that we follow exactly the proposed algorithm of the previously mentioned paper.
The first step (see subection 4.1) is to replace the bottom with its continuous,
 piecewise linear approximation and this even if the bottom is already continuous.
'''

############################# the Piecewise linear approximation of the bottom#######
# the bottom at the x_j=jdx j=0,...Nx and at x_j=(j+1/2) , j=-1, ...Nx
# see the file : needed..
BLC,Batxj=getPiecewise_l_Bottom(Nx,dx,L)
 

###############################   initial data       #######################
## go to file Bottom_... to change the data at t=0, you will find it at the end of the file
h,q=inital_dat(Nx,Batxj)

########################## force the boundary conditions at t=0 ##################
# see Bottom_...
boundary_cd(0,h,q)



########################### Call the solvers, chose either Forward Euler solver #################
###########################     or SSP-RK  for the time update                  ################
# go to neededfunction... file

'''
#Forward Euler solver
update_timeFE(T,h,q,Batxj,Nx,BLC,g,dx)
'''

#third-order strong stability preserving Runge-Kutta
h_history,q_history,time_history,dt_history=update_timeRK3(T, h, q, Batxj, Nx, BLC, g, dx)


'''
At this stage, the reader must understand that the direct solver has been applied 
and now you have the approximate surface and discharge.
 If any problem occurs, check the given initial and boundary data.
Also, do not apply a non-homogeneous Neumann condition in the code:
 because the direct solver here cannot handle this case. However
 if it is needed, one can modify the ghost points in the direct solver code 
 . Moreover,Sometimes you will get an error simply because the bottom 
is strictly higher than the initial surface. Even though the code addresses the dry
 state, you must not set the distance between the bottom and the surface to be very small
(e.g., 1e-10).
'''


'''
 one can always email me
 noureddine.lamsahel@um6p.ma, for any further information

'''



###################### Plot the output data at a chosen instant ############

##Here we plot the results of the direct solver, namely the surface and the discharge.
# Choose the instant.
indx_t_g=-1                                 # T the final time 
t_g=time_history[indx_t_g]
h_t_g=h_history[indx_t_g]
q_t_g=q_history[indx_t_g]

# A function that will plot eta B and q at the instant t_g
#plot_given_time(X,h_t_g,Batxj,q_t_g,t_g)

'''
Here we  generate an MP4 video of the plots. After creating the video, 
close the current console; otherwise, you will not be able to view the next simulation plots.
For this, we first save all the plots, then we generate the video, then we close the console
before running your next simulation.
'''

nt,ns=np.shape(h_history)
h_historyv = np.asarray(h_history)
b_vec      = np.asarray(Batxj)
#video_plot(X,nt,b_vec,h_historyv,time_history)









#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ The inverse problem solver \\\\\\\\\\\\\\\\\\\\\\\\\\\
#################################                              ########################
'''
In this part, there are typically two main sections:
In the first section, we handle the steady state, presenting an exact solver that requires
less regularity on the surface.
In the next section, we address the unsteady case using an  approximate solver 
to obtain an approximation of the bottom.

'''


########################### solver for the steady state ################################

'''
################## the analytic solution for a steady state flow
indx_t_g_inv=-1  
t_g=time_history[indx_t_g_inv]
q_end=q_history[indx_t_g_inv] 
eta_end=Batxj+h_history[indx_t_g_inv]         #h=eta-b

B_analytic=exact_solver(q_end,eta_end,Nx,g,L)
plot_analyticsolution(X,h_t_g,Batxj,B_analytic,t_g)
print("////////////////////////////////////////////////////////////////////////")
print('Relative L^inf error for anayltic solution:')
print(f'{100* np.linalg.norm(Batxj-B_analytic, ord=np.inf)/ np.linalg.norm(Batxj, ord=np.inf)} %')

print('Relative L^2  error for anayltic solution:')
print(f'{100* np.linalg.norm(Batxj-B_analytic, ord=2)/ np.linalg.norm(Batxj, ord=2)} %')
'''







########################  solver for unsteady state ######################


##from eta( the free surface) and the boundary data q_0 
# Notation :in the our paper we denoted zeta for the surface, here we use eta following the original paper 
'''
 This function constructs q from the free surface over time and the q(t,0) by solving
q_x = –η_t, using Heun’s method for space and a nonuniform central 
three-point finite-difference scheme for time.
'''
q_constucted=compute_q_from_surface_Bc(h_history,Batxj,dt_history,time_history,dx)

'''
After having q from the surface and the Bc data, we look for the times for which q > beta.
 For the moment, beta = 1e-2
'''
indx_q_pos=find_indes_q_str_positive(q_constucted)

################ Call for the inverse problem solvers: first- and second-order bottom approximation
# We first choose the index and time for q from the part where q > beta
indx_over_positive_inds=-1
indx_resp_q=indx_q_pos[indx_over_positive_inds]  
t_g=time_history[indx_resp_q]
#print(f'time={t_g}')

# Free surface at this instant 
h_t_g=h_history[indx_resp_q]
eta_n=Batxj+h_t_g

######## if  noisy free surface 
'''
Here is the case when we have a noisy surface, exactly Test 5 in our the paper.
 Since the surface is noisy, we must first smooth it in order to have a reasonably 
 small Lipschitz constant.
'''


'''

p=0.01        # 2% noise as the depth is 2
eps = np.random.uniform(-p, p, size=Nx+1)
# add noise
eta_noisy=eta_n+eps*h_t_g

smoothing_factor = 0.011  #  always small to avoid losing the shape

# Build spline cubic
spline = UnivariateSpline(X, eta_noisy, k=3, s=smoothing_factor)

# Evaluate the smoothed profile on the same x
eta_smooth =spline(X)

eta_n=eta_smooth

'''

########Here we start calling the solvers
'''
Here we propose two solvers:
The first solver is a first order solver that uses Forward Euler to approximate eta_x and also 
to solve the inverse system.
The second solver is a second order solver that uses three-point central finite differences to approximate eta_x
 and the second-order Heun’s RK method to solve the inverse system.

'''

   # First order solver:
#B_appx_inv=solver_inver_problem_FD(indx_resp_q,q_constucted,eta_n,dx,g,L,Nx,dt_history,time_history)

   #second order solver
B_appx_inv=solver_inver_problem_FD_scondeorder(indx_resp_q,q_constucted,eta_n,dx,g,L,Nx,dt_history,time_history,X)



'''
Now that we have the approximate bottom from the surface data, we plot the exact
 and approximate bottoms as well as the exact and approximate discharges
'''


#Plot the bottom, the surface, then the bottom and the approximate bottom
plot_given_time_inv(X,h_t_g,Batxj,B_appx_inv,t_g)

# Plot direct solver discharge and inverse problem discharge
plot_directDisch_inverseDisch(q_constucted[indx_resp_q],q_history[indx_resp_q],t_g,X)




#///////////////////////////////// # Check the two sufficient conditions (12) and (24) in our paper #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\for q > beta

###################################                                               ###########################

print('/////////////////////////////////////////////////////////////////')
print('Here we Compute C_1 and C_2 for Lamda_1 and Lamda_2')

t_far_zero=0  # To be as far as possible from the instant t = 0
ts,ss=np.shape(q_constucted)
min1l=[]
max2l=[]
for t in range(t_far_zero,ts):
    q_ts=q_history[t]
    h_tsl=h_history[t]
    h_ts=np.asarray(h_tsl)
    
    #compute Lamda_1
    min1=min(q_ts/h_ts+np.sqrt(g*h_ts))  # min over x fro this instant
    min1l.append(min1)
    
    #compute Lamda_2
    max2=max(q_ts/h_ts-np.sqrt(g*h_ts))  #max over x for the current instant
    max2l.append(max2)

C_1=min(min1l)
C_2=-max(max2l)
print(f' C_1 form Lamda_1={C_1}')
print(f' C_2 form Lamda_2={C_2}')

#plot the speeds
plotlamda1andlamda2(time_history[t_far_zero:],min1l,max2l)



# Value of the bottom variation for our conditions
sum_part_BOttom=0.6  

##left and right info

'''
print('///////////////////////////////////////////////////////////////')
print('here we verify the conditions needed for the satisfaction of the q>0')
print('///////////////////////////////////////////////////////////////')
print('Here using left and right information')

for t1 in range(t_far_zero,ts):
    for t2 in range(t_far_zero,ts):
        q_ts1=q_history[t1]
        h_tsl1=h_history[t1]
        h_ts1=np.asarray(h_tsl)
        U1=q_ts1/h_ts1
        
        q_ts2=q_history[t2]
        h_tsl2=h_history[t2]
        h_ts2=np.asarray(h_tsl)
        U2=q_ts2/h_ts2
        leftt=U1[0]+U2[-1]
        rightt= g*( 1/C_01 +  1/C_0 )*sum_part_BOttom  - 2*np.sqrt(g)*(np.sqrt(h_ts1[0])-np.sqrt(h_ts2[-1]))  
        if leftt<=rightt:
            print(time_history[t1])
            print('no')
    


print('End of Left and right info now only one side')
print('///////////////////////////////////////////////////////////////')
print('Here using left information only')

eta_Time_max=[]
for t in range(t_far_zero,ts):
    h_t=h_history[t]
    eta_t=np.asarray(h_history[t])+Batxj
    eta_tmax=max(eta_t)
    eta_Time_max.append(eta_tmax)

max_eta=max(eta_Time_max)   

print("now we verifty the left info condition in the fast way")
for t1 in range(t_far_zero,ts):
    #compute the inlet velocity for t1
    q_ts1=q_history[t1]
    h_tsl1=h_history[t1]
    h_ts1=np.asarray(h_tsl1)
    U1=q_ts1/h_ts1
    
    #for h at time t2 compute the max of the surface
    #left in the formula
    leftt=U1[0]
    #right in the formula
    rightt=-2*np.sqrt(g)*( np.sqrt(h_ts1[0])-np.sqrt( max_eta-min(Batxj) ) ) +(g/C_0)*sum_part_BOttom
    
    if leftt<=rightt:
        print(time_history[t1])
        print('no')
   # else:
       
       
'''  
'''
print('/////////////////////////////////////////////////////////////////////////')
print('We compute the Froude number')         
Fr_n=np.empty((ts,ss))  
for t in range(ts):
    q_t=q_history[t]
    h_tl=h_history[t]
    h_t=np.asarray(h_tl)
    u_t=q_t/h_t
    for j in range(ss):
        Fr_n[t,j]=u_t[j]/np.sqrt(g*h_t[j])
        #if Fr_n[t,j]<1:                #and time_history[t]>40
         #   print(time_history[t])
         #   print('out')
       # elif Fr_n[t,j]==1:
        #    print(time_history[t])
        #    print(' critical')
       # else:
       #    print(time_history[t])
       #    print('subcritical')


#plot  Fr at a time 
plotFr(Fr_n[-1,:],X)

'''
    
########## Here we compute the relative errors
print("////////////////////////////////////////////////////////////////////////")
print('Relative L^inf error:')
print(f'{100* np.linalg.norm(Batxj-B_appx_inv, ord=np.inf)/ np.linalg.norm(Batxj, ord=np.inf)} %')

print('Relative L^2  error:')
print(f'{100* np.linalg.norm(Batxj-B_appx_inv, ord=2)/ np.linalg.norm(Batxj, ord=2)} %')



'''
In the case there is noise in the surface, 
the following will plot the graphs. However, note that we should first stop the non-noisy case plots;
 otherwise we will generate the same plots, but the first ones will not have the correct labels,
 even though the graphs are the same.
'''

#plot_noisycase(X,Batxj,h_t_g,eta_noisy,t_g,eta_smooth,B_appx_inv)
