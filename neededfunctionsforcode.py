#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:10:31 2025

@author: lamsahel
"""
import numpy as np
from Bottom_inital_BCdatas import *

'''
# This file contains every function needed for the direct solver from A to Z. 
At the end, the user has two options for the time update: a simple solver using
 Forward Euler or the solver proposed in the paper: the third-order strong stability-preserving 
 Runge–Kutta (SSP-RK).


'''






#  Piecewise linear approximation of the bottom: computation of B_{j+1/2} and B_j
def getPiecewise_l_Bottom(Nx,dx,L):
    BLC = np.zeros(Nx+2)      #for points (j+1/2)dx j=-1, .....,NX
    Batxj=np.zeros(Nx+1)      #for points jdx 
    
      # discontinuous bottom (or even continuous)
    for j in range(1, Nx+1):
       #eps=dx/2                    #  for example
       epsl=1e-7
       x_j12=(j-0.5)*dx
       BLC[j]=( bottom_f(x_j12+epsl)+bottom_f(x_j12-epsl) )/2.         #comput B_{j+1/2} see (36)
     
    BLC[0]= bottom_f(-dx/2)
    BLC[-1]= bottom_f(L + dx/2)

    for j in range(0, Nx+1):
        Batxj[j]=0.5*(BLC[j]+BLC[j+1])  
    
    return BLC,Batxj
    
    



def minmod_f(a, b, c):
    #minmod function (41) 
    if a>0 and b>0 and c>0:
        return min(a,b,c)
    elif a<0 and b<0 and c<0:
        return max(a,b,c)
    else:
        return 0.0

 

#Here we apply ghost points to enforce the boundary conditions in the finite-volume method

def apply_gostpoint(ex_w,ex_q):
    ex_q[0]=ex_q[1]
    ex_q[-1]=ex_q[-2]
    
    ex_w[-1]=ex_w[-2]
    ex_w[0]=ex_w[1]


# Here we compute U_x (equation 41)
def compute_div_ofU(ex_w,ex_q,N):
    
    # Theta used to control the amount of numerical viscosity in the resulting scheme.
    #For the moment, we choose theta = 1 as in the paper tests.

    theta=1.
    div_w = np.zeros(N+1)
    div_q = np.zeros(N+1)

    for j in range(1, N+2):
        #derivative of the free surface 
        div_w_L = ex_w[j]   - ex_w[j-1]
        div_w_C = ex_w[j+1] - ex_w[j-1]
        div_w_R = ex_w[j+1] - ex_w[j]
        div_w[j-1] = minmod_f(theta*div_w_L, 0.5*div_w_C, theta*div_w_R)
        
    for j in range(1, N+2):
        #derivative of the flow rate
        div_q_L = ex_q[j]   - ex_q[j-1]
        div_q_C = ex_q[j+1] - ex_q[j-1]
        div_q_R = ex_q[j+1] - ex_q[j]
        div_q[j-1] = minmod_f(theta*div_q_L, 0.5*div_q_C, theta*div_q_R)
        
        
    # Observe that we do not divide by dx, as afterwards we will multiply by dx

    return  div_w,div_q


'''
we mentioned in our work, for brevity we omitted the details for the positivity preserving in the paper; 
however, the code is written for positivity preserving with the details. 
We refer the reader of this code to the main reference of this file to understand the steps.

'''

def keeppostiveh(h_j,hu_j,dx):
    eps_val=(dx)**4
    valkeep=( np.sqrt(2)*h_j*hu_j )/np.sqrt( h_j**4+max(h_j**4,eps_val) )
    return valkeep




'''
 Here we present the main part of the code for the forward solver, which corresponds
 to the space march of the code (34)
'''
def compute_RHS(h, q, Batxj,N,BLC,g,dx):
    # We use the notation of the ref paper
    w = h + Batxj       #  is exactly zeta in our paper 
    ex_q=np.zeros(N+3)
    ex_q[1:-1]=q
    ex_w=np.zeros(N+3)
    ex_w[1:-1]=w
    
    ##### at the gost point
    apply_gostpoint(ex_w,ex_q)
    
    # recall that U = (w, q), where q = h u
    div_w,div_q=compute_div_ofU(ex_w,ex_q,N)
    
    # here H_div is to put the elemnt H_{j+1/2} - H_{j-1/2}, see equation (34)
    #H_divw=[]
    H_divw=np.zeros(N+1)
    
    #H_divq=[]
    H_divq=np.zeros(N+1)
    a_CFL=[]
    for j in range(0,N+1):     #j from 0 to N_x
    
      #we first look at j+1/2 and j-1/2 and compute H_{j±1/2}

         #j+1/2
        #right U at j+1/2
        if j==N:
            w_R1=ex_w[-1]
            q_R1=ex_q[-1]
        else:
           w_R1=w[j+1] - 0.5*div_w[j+1]
           q_R1 = q[j+1] - 0.5*div_q[j+1]
           
        #lift U at j+1/2
        w_L1 = w[j]   + 0.5*div_w[j]
        q_L1 = q[j]   + 0.5*div_q[j]
        
         ###j-1/2
        
        #right U at j-1/2
        w_R=w[j] - 0.5*div_w[j]
        q_R = q[j] - 0.5*div_q[j]
        
        #lift U at j-1/2
        if j==0:
            w_L=ex_w[0]   #= at 1
            q_L=ex_q[0]    # = at 1
        else:
            w_L = w[j-1]   + 0.5*div_w[j-1]
            q_L = q[j-1]   + 0.5*div_q[j-1]
          
        

#At this line of the code, we have U^± at j+1/2 and j-1/2. Now we compute h^± at these points.
#In the notation, we add '1' when at j+1/2 and nothing when at j-1/2.
#'L' stands for '-' and 'R' for '+'.

         
       ####j+1/2 
            
        h_R1=w_R1-BLC[j+1]
        h_L1=w_L1-BLC[j+1]
        ##################################j-1/2 
        h_R=w_R-BLC[j]
        h_L=w_L-BLC[j]
        
      ## Positivity preserving
        
        if h_L1<0:
            w_L1=BLC[j+1]
            w_R1=2*w[j]-BLC[j+1]
            
            h_R1=w_R1-BLC[j+1]
            h_L1=0.
    
        
        if h_R<0:
            w_R=BLC[j]
            w_L=2*w[j]-BLC[j]
            
            h_L=w_L-BLC[j]
            h_R=0.
        
        ### Compute the flow rate q± and u± when the h± are small (see (2.17))

        
        #########j+1/2
        u_R1=keeppostiveh(h_R1,q_R1,dx)
        u_L1=keeppostiveh(h_L1,q_L1,dx)
        #########j-1/2
        u_R=keeppostiveh(h_R,q_R,dx)
        u_L=keeppostiveh(h_L,q_L,dx)
        
        ######### Recompute q as in (2.21)

        
        #########j+1/2
        q_R1= h_R1*u_R1
        q_L1= h_L1*u_L1
        
        #########j-1/2
        q_R= h_R*u_R
        q_L= h_L*u_L
        
        ######### Compute a^+- the local speeds of propagation  see (2.22) and (2.23)
        # see also (42) in our work
        #########j+1/2
        lamda_R1=np.sqrt(g*h_R1)
        lamda_L1=np.sqrt(g*h_L1)
        
        a_R1=max(u_R1+lamda_R1,u_L1+lamda_L1,0)
        a_L1=min(u_R1-lamda_R1,u_L1-lamda_L1,0)
        
        #########j-1/2
        lamda_R=np.sqrt(g*h_R)
        lamda_L=np.sqrt(g*h_L)
        a_R=max(u_R+lamda_R,u_L+lamda_L,0)
        a_L=min(u_R-lamda_R,u_L-lamda_L,0)
        
        ###################Compute   F(w,q)=(q,qu+0.5*gh^2) see (2.3)
        #########j+1/2
        Fw_R1=q_R1
        Fw_L1=q_L1
        
        Fq_R1=q_R1*u_R1+0.5*g*h_R1*h_R1
        Fq_L1=q_L1*u_L1+0.5*g*h_L1*h_L1
        
        #########j-1/2
        Fw_R=q_R
        Fw_L=q_L
        
        Fq_R=q_R*u_R+0.5*g*h_R*h_R
        Fq_L=q_L*u_L+0.5*g*h_L*h_L
        
        ###### Compute H see (2.11)
        #########j+1/2
        dom_a1=a_R1-a_L1
        H_w1=(  a_R1*Fw_L1-a_L1 *Fw_R1  )/dom_a1+ a_R1*a_L1*(w_R1-w_L1  )/dom_a1
        H_q1=(  a_R1*Fq_L1-a_L1 *Fq_R1  )/dom_a1+ a_R1*a_L1*(q_R1-q_L1  )/dom_a1
        
        #########j-1/2
        dom_a=a_R-a_L
        H_w=(  a_R*Fw_L-a_L *Fw_R  )/dom_a+ a_R*a_L*(w_R-w_L  )/dom_a
        H_q=(  a_R*Fq_L-a_L *Fq_R  )/dom_a+ a_R*a_L*(q_R-q_L  )/dom_a
        
        
        #Final steps, see for example (2.24) and (2.2) or our paper (38) and (39)
        H_wdif=-(H_w1-H_w)/dx
        H_divw[j]=H_wdif
        
        S_j=-g*h[j]*(BLC[j+1]-BLC[j])/dx   
        
        H_qdif=-(H_q1-H_q)/dx +S_j
        H_divq[j]=H_qdif
        
        
        #####for CFL see Thm 2.1
        a_CFL.append(max(a_R1,-a_L1))
    
        
    return H_divw,H_divq, max(a_CFL)
        
      
        
      
        
'''    
We have the space march of the code; now we turn to the time march.
 We propose two solvers
'''
 
    
 ###################  Here is a Forward Euler solver 
def update_timeFE(T,h,q,Batxj,N,BLC,g,dx):
    CFL = 0.9    ## Just to control the time in the formula of Theorem 2.1
    
    # Let's start
    t=0.
    
    # enforce BCs at t=0
    boundary_cd(t,h,q)
    w=h+Batxj
    while t < T:
        print(t)
        
        H_divw,H_divq,a=compute_RHS(h, q, Batxj,N,BLC,g,dx)
        
               # As Thm 2.1 see also our paper (43)
        dt=CFL*dx/(2*a)
        
        # To ensure that we reach the end

        if t + dt > T:
            dt = T - t
            
        w[:]=w+dt* H_divw
        h[:]=w-Batxj
        
        q[:]=q+dt*H_divq
        boundary_cd(t+dt,h,q)
        
        t += dt
       

##### third-order strong stability preserving Runge-Kutta (SSP-RK)
        
        
def update_timeRK3(T, h, q, Batxj, N, BLC, g, dx):
   # Here we use the time solver proposed in the reference paper
   # As in the Euler case, we start by enforcing the boundary conditions
   # This is important when we have some discontinuous initial data at the boundaries

    CFL = 0.9
    t = 0.
    # enforce BCs at t=0
    boundary_cd(t, h, q)
    w=h+Batxj
    
    ##store the time update for h and q
    h_history = [h.copy()]
    q_history=[q.copy()]
    dt_history=[]
    
    
    # As the time step changes, it isimportant to know each index and its associated time
    time_history=[0]
    while t < T:
        print(t)
        # compute RHS and max wave‐speed a_max (Thm 2.1)
        H_divw, H_divq, a_max = compute_RHS(h, q, Batxj, N, BLC, g, dx)
        dt = CFL * dx / (2 * a_max)
        if t + dt > T:
            dt = T - t
        
        
        ## the method have 3 stages 
            # Stage 1 : U^1=U^n +dt H^n_j(U^n)
        w1 = w   + dt * H_divw
        h1=w1-Batxj
        
        q1 = q   + dt * H_divq
        boundary_cd(t + dt, h1, q1)
        
       

            # stage 2 :U^2= 3/4 U^n+1/4 *( U^1+ dt* H^n_j(U^1)   )
        H_divw1, H_divq1, _ = compute_RHS(h1, q1, Batxj, N, BLC, g, dx)
        w2 = (3./4.)*w   + 0.25*(w1 + dt * H_divw1)
        h2=w2-Batxj
        
        q2 =(3./4.)*q   + 0.25*(q1 + dt * H_divq1)
        boundary_cd(t + 0.5*dt, h2, q2)
        
        

             # stage 3 : U^{n+1}= 1/3 U^n +(2/3) ( U^2+dt H^n_j( U^2) )
        H_divw2, H_divq2, _ = compute_RHS(h2, q2, Batxj, N, BLC, g, dx)
        w[:] = (1./3.)*w + (2./3.)*(w2 + dt * H_divw2)
        h[:]=w-Batxj
        
        q[:] = (1./3.)*q + (2./3.)*(q2 + dt * H_divq2)
        boundary_cd(t + dt, h, q)
        
        # Store the time update of h and q
        # at time t
        h_history.append(h.copy())
        q_history.append(q.copy())
        
        # store the time
        time_history.append(t+dt)
        
        #get dt update
        dt_history.append(dt)
        
        #update time
        t += dt

    return h_history,q_history,time_history,dt_history
        
        
        
        
        
 
'''
The reader can update the code; for example, it is possible to accelerate it or modify functions 
to reduce CPU time or memory. However, without any modification to the code,
 it is relatively  fast and stable for  the 1d case considered in our work.
'''                         
      
  
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    