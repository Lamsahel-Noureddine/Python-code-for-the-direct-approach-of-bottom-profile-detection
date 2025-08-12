#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 20:59:12 2025

@author: lamsahel
"""


import numpy as np
from Bottom_inital_BCdatas import bottom_f,q_inlet_Bc


#########################  Exact solver for steady state #####################
def exact_solver(q_end,eta_end,Nx,g,L):
    B_0=bottom_f(0)
    B_app=[B_0]
    H_0=eta_end[0]-B_0
    q_canstant=q_end[0]

    for j in range(1,Nx+1):
        dom_h=1./(H_0*H_0) - ( 2.0*g/(q_canstant*q_canstant)  )*( eta_end[j]-eta_end[0])
       
        if dom_h<=0:
            print(j)
            print(dom_h)
            print( eta_end[j]-eta_end[0])
            
        else:
            h_j2=np.sqrt(1./dom_h)
            B_app.append(eta_end[j]-h_j2)
    return  B_app




#s##################s####solver for the unsteady state ##############"

 ###################Compute q from the surface and the boundary data ###########
    ''' 
 We first create a function that computes the time derivative.
This function uses the extrapolation at the end point and the central 3 points FD
'''
def approximation_time_der(f_list, dt_history,time_history):
    # Second order app of f_t on a nonuniform time grid dt_history
    #introduce ghost points t_{-1} and t_{N+1}, where t_N=T
     #  nonuniform central for t_n where 0<=n<=N 

     
    #just to force it to array
    func = np.asarray(f_list)
    dt  = np.asarray(dt_history)
    
    #the size of the data
    tz,sz=np.shape(func)

    # Reconstruct original time grid t=[0..tz-1] from t_history and force to array
    t = np.asarray(time_history)


    #  Build extended grid with ghost points at 0 and T
    t_ext = np.empty(tz + 2)
    t_ext[1:-1] = t
    t_ext[0]     = t[0]    - dt[0]    # left ghost time
    t_ext[-1]    = t[-1]   + dt[-1]   # right ghost time

    # Extend function array
    f_ext = np.zeros((tz + 2, sz))
    f_ext[1:-1] = func

    # Extrapolate left ghost via quadratic Lagrange to get
    # an approximation of f(t_{-1})from (t0,f0),(t1,f1),(t2,f2)
    t0=t_ext[1]            #0
    t1=t_ext[2]            #dt_0
    t2 = t_ext[3]          # dt_0+dt_1
    tau_left   = t_ext[0]  #t_{-1}
    
    for i in range(sz):
        #the values at the right point of zero
        f0=func[0,i]
        f1=func[1,i]
        f2 = func[2,i]
        
        #compute the Lagrange quadratic at the point t_{-1}
        L0 = (tau_left - t1)*(tau_left - t2)/((t0 - t1)*(t0 - t2))
        L1 = (tau_left - t0)*(tau_left - t2)/((t1 - t0)*(t1 - t2))
        L2 = (tau_left - t0)*(tau_left - t1)/((t2 - t0)*(t2 - t1))
        
        #approximate the function at t_{-1}
        f_ext[0,i] = f0*L0 + f1*L1 + f2*L2

    # Extrapolate right ghost via quadratic Lagrange 
    # to get an approximation of f(t_{N+1}) from (t_{N-2},f_{N-2}),(t_{N-1},f{N-1}),(tN,fN)
    tNm2=t_ext[-4]     #notation as before
    tNm1=t_ext[-3]
    tN =  t_ext[-2]
    tau_right     = t_ext[-1]
    for i in range(sz):
        fNm2=func[-3,i]
        fNm1=func[-2,i]
        fN =  func[-1,i]
        
        M2 = (tau_right - tNm1)*(tau_right - tNm2)/((tN - tNm1)*(tN - tNm2))
        M1 = (tau_right - tN)*(tau_right - tNm2)/((tNm1 - tN)*(tNm1 - tNm2))
        M0 = (tau_right - tN)*(tau_right - tNm1)/((tNm2 - tN)*(tNm2 - tNm1))
        
        f_ext[-1,i] = fNm2*M0 + fNm1*M1 + fN*M2

    #  Compute central 3‐point nonuniform derivative on extended grid
    dt_ext=np.zeros(tz+1)
    dt_ext[1:-1]=dt
    dt_ext[0]=dt[0]
    dt_ext[-1]=dt[-1]
   
    D_ext  = np.zeros_like(f_ext)  # to compute the derivative
    for n_ext in range(1, tz+1):
        delta_m=dt_ext[n_ext-1] # t_n-t_{n-1}
        delta_p =  dt_ext[n_ext] #t_ {n+1}-t_{n}
        
        denom_n =delta_m * delta_p * (delta_m + delta_p )
        D_ext[n_ext] = (  (delta_m**2)*f_ext[n_ext+1] - (delta_p**2)*f_ext[n_ext-1]  + (delta_p**2 - delta_m**2)*f_ext[n_ext] ) / denom_n

    return D_ext[1:-1]
 


##### This function solves q_x = -eta_t to get the q data from eta and the inlet discharge

def compute_q_from_surface_Bc(h_history,Batxj,dt_history,time_history,dx):
   # Given the surface data (h_history + Batxj) from the direct approach,
   # the distance dx between observed surface measurements,
   # and the boundary data at x = 0 for q, q(0, t)

    
    # we aim to solve q_x(t,x)=-eta_t(t,x)
    tz, sz = np.shape(h_history)
    q_constucted=np.zeros((tz,sz))    # tz time size, sz space size
    
    #Compute the eta_t using the given dt_history and h_history
    Deta = approximation_time_der(h_history, dt_history,time_history) # Remark that eta_t=h_t
    
    #Second-order accurate reconstruction of 
    #here we use Heun's method ( improved Euler)
    # the formula is simplifid sinse eta is not a function on q
    for n in range(tz):
        time_n=time_history[n]
        #compute the values at the inlet
        q_constucted[n,0]=q_inlet_Bc(time_n)
        for i in range(sz-1):
            #predictor step
            f_i   = -Deta[n,i]
            #corector step
            f_ip1 = -Deta[n,i+1]
            #the result in two stage (RK)
            q_constucted[n,i+1] = q_constucted[n,i] + (dx/2)*(f_i + f_ip1)
    return q_constucted


 

     #########################""find the index   where q_constructed>beta ######
def find_indes_q_str_positive(q_constucted):
    tz, sz = np.shape(q_constucted)
    #the lower-bound
    beta_chosing=1e-2
    # to take the values 
    indx_q_pos=[]
    for n in range(tz):
        if min(q_constucted[n,:])>=beta_chosing:
            indx_q_pos.append(n)
            
        
    
    return indx_q_pos

       ###################### find the maximum value far from zero and the corresponding index

def find_the_large_value(indx_q_pos, values_q_pos):
    Large_index=indx_q_pos[0]
    Large_valu=values_q_pos[0]
    
    N_p=np.size(indx_q_pos)
    for i in range(N_p):
        ind=indx_q_pos[i]
        Large_valu_ind=values_q_pos[i]
        if min(Large_valu)<=min(Large_valu_ind):
            Large_valu=Large_valu_ind
            Large_index=ind
        
    return Large_index,Large_valu
    
'''   
Now we have everything, we start solving the inverse equation at a given time where q > beta 
using either the first- or second-order solver
'''
    
# first order solver
def solver_inver_problem_FD(time_index,q_constucted,eta_n,dx,g,L,Nx,dt_history,time_history):
    # Given the instant of observation
    # Given the reconstructed q from the surface and the inlet data
    # Given the free surface
   
    
    # Left boundary data

    B_0=bottom_f(0)
    B_app=[B_0]
    H_01=eta_n[0]-B_0
    H_0=[H_01]
    
    
    # Take the value of q at this time, q(t*, x)
    q_n=q_constucted[time_index]
    
    # Compute the time derivative of q for all t
    Dq= approximation_time_der(q_constucted, dt_history,time_history)
    
    # Take the derivative at the chosen observation time, q_t(t*, x)
    dq_time=Dq[time_index]
    
    for i in range(0,Nx):
        dom_h=( q_n[i]**2 / H_0[i]  )  - g*( eta_n[i+1] - eta_n[i] )* H_0[i] - dx*dq_time[i] 
      
        if dom_h<=0:
           print( 'Error Left')
           print(i)
           
           
        else:
           H_0.append(q_n[i+1]**2 / dom_h)
           #print(H_0[i+1])
           B_app.append(eta_n[i+1]-H_0[i+1])
        
    return B_app
    
     
    


#### the second arder solver


def solver_inver_problem_FD_scondeorder(time_index,q_constucted,eta_n,dx,g,L,Nx,dt_history,time_history,X):
    # Given the instant of observation
    # Given the reconstructed q from the surface and the inlet data
    # Given the free surface
   
    
    # Left boundary data
    B_0=bottom_f(0)
    B_app=[B_0]
    H_01=eta_n[0]-B_0
    H_0=[H_01]
    

    
    
    # Take the value of q at this time, q(t*, x)
    q_n=q_constucted[time_index]
    
    # Compute the time derivative of q for all t
    Dq= approximation_time_der(q_constucted, dt_history,time_history)
    
    # Take the derivative at the chosen observation time, q_t(t*, x)
    dq_time=Dq[time_index]
    
    # Compute the spatial derivative of eta_n
    dx_list = np.full(Nx, dx)
    eta_resp=eta_n.reshape(-1, 1)
    eta_x=approximation_time_der(eta_resp,dx_list ,X).flatten()
    
    zeta = np.empty_like(q_n)
    zeta[0] = q_n[0]**2 / H_01
    
    # march in x with Heun's method for zeta_x


    for i in range(Nx):
        rhs1 = -g * eta_x[i] * (q_n[i]**2 / zeta[i]) - dq_time[i]
        u_star = zeta[i] + dx * rhs1

        # Corrector
        rhs2 = -g * eta_x[i+1] * (q_n[i+1]**2 / u_star) - dq_time[i+1]
        # Heun update
        zeta[i+1] = zeta[i] + (dx/2) * (rhs1 + rhs2)
        if zeta[i+1]<=0:
           print( 'Error Left')
           print(i)
        else:
           H_0.append( q_n[i+1]**2 / zeta[i+1])
           B_app.append(eta_n[i+1]-H_0[i+1])
           
    return B_app
 
    
 




   
    