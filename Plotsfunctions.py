#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 19:45:36 2025

@author: lamsahel
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib

'''
This file contains all functions that help plot the results of the direct and inverse solvers.
'''

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Direct solver _\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_
################################                   ####################################
def plot_given_time(X,h_t_g,Batxj,q_t_g,t_g):
        
    ###now plot for the surface and the bottom
    plt.figure(figsize=(8,4))
    plt.plot(X,h_t_g+Batxj,color='blue',linestyle='-', label=r'$\zeta$')
    plt.plot(X,Batxj,color='black',linestyle='-',marker='o',markerfacecolor='none', label=r'$b$')    # if its a problem remove lm : size of the line graph
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'$t_f\;$={t_g} s ')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("surface_and_bottom.png")
    plt.savefig("surface_and_bottomq",format="pdf", bbox_inches="tight" )
    plt.show()
   
    ###now plot the disharge q
    plt.figure(figsize=(8,4))
    plt.plot(X,q_t_g,color='blue',linestyle='None',marker='x',markersize=3., label=r'$q$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'$t_f\;$={t_g} s')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("discharge q")
    plt.savefig("discharge q",format="pdf", bbox_inches="tight" )

    plt.show()
    
    
def video_plot(X,nt,b_vec,h_historyv,time_history):

    matplotlib.use("Agg")
    timing=25
      #frames: each 20 we plot the domain
    frames = list(range(0, nt,  timing))
    if frames[-1] != nt-1:
        frames.append(nt-1)

        # seting of the figure
    fig, ax = plt.subplots()
    line_num, = ax.plot([], [], 'b-', label='$\\eta$')
    line_bottom, = ax.plot([], [], color='black',linestyle='-',lw=2.5, label='$B$')
    ax.set_xlim(X.min(), X.max())
    vals = np.vstack([h_historyv[frames, :] + b_vec, b_vec])
    ax.set_ylim(vals.min(), vals.max() + 0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('height')
    ax.legend()

    def update(t):
        line_num.set_data(X, h_historyv[t, :] + b_vec)
        line_bottom.set_data(X, b_vec)
        timplot=time_history[t]
        ax.set_title(f"t = {timplot:.2f}")

    
    update(frames[0])
    fig.canvas.draw()

    # configure ffmpeg    
    #  if the video is not done then you must install ffmpeg
    writer = FFMpegWriter(
        fps=4,
        codec='libx264',
        bitrate=3000,
        metadata={'artist': 'Vous'}
    )

    #loop and draw
    with writer.saving(fig, "animation.mp4", dpi=200):
        for t in frames:
            update(t)
            fig.canvas.draw()         
            fig.canvas.flush_events()  
            writer.grab_frame()
            

    plt.close(fig)
    print("Done")
    




#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ inverse solver \\\\\\\\\\\\\\\\\\\\\_
#######################################                  ###########################

# Function for the analytic solution case
def plot_analyticsolution(X,h_t_g,Batxj,B_appx,t_g_inv):
        
    ###Plot the surface and the bottom and the bottom appr
    plt.figure(figsize=(8,4))
    plt.plot(X,h_t_g+Batxj,color='blue',linestyle='-',marker='x', label=r'$\zeta$')
    plt.plot(X,Batxj,color='black',linestyle='-',lw=2.5, label=r'$b$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,B_appx,color='red',linestyle='none',marker='o',markerfacecolor='none', label=r'$b\;\mathrm{analytic}$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f' $t_f\;$={t_g_inv} s')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("surface_and_bottom_bottomapp.png")
    plt.savefig("surface_and_bottom_bottomappanalytic.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
   ###Plot the bottom and the bottom appr
    plt.figure(figsize=(8,4))  
    plt.plot(X,Batxj,color='black',linestyle='-',lw=2.5, label=r'$b$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,B_appx,color='red',linestyle='none',marker='o',markerfacecolor='none', label=r'$b\;\mathrm{analytic}$')
    #plt.title('Bottom and the approximate Bottom ')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("bottom_bottomapp.png")
    plt.savefig("bottom_bottomappanalytic.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
  
    
def plot_given_time_inv(X,h_t_g,Batxj,B_appx,t_g_inv):
        
    ###Plot the surface and the bottom and the bottom appr
    plt.figure(figsize=(8,4))
    plt.plot(X,h_t_g+Batxj,color='blue',linestyle='-',marker='x', label=r'$\zeta$')
    plt.plot(X,Batxj,color='black',linestyle='-',lw=2.5, label=r'$b$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,B_appx,color='red',linestyle='none',marker='o',markerfacecolor='none', label=r'$b\;\mathrm{reconstructed}$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f' $t_f\;$={t_g_inv} s')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("surface_and_bottom_bottomapp.png")
    plt.savefig("surface_and_bottom_bottomapp.pdf",format="pdf", bbox_inches="tight" )

    plt.show()
    
   ###Plot the bottom and the bottom appr
    plt.figure(figsize=(8,4))  
    plt.plot(X,Batxj,color='black',linestyle='-',lw=2.5, label=r'$b$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,B_appx,color='red',linestyle='none',marker='o',markerfacecolor='none', label=r'$b\;\mathrm{reconstructed}$')
    #plt.title('Bottom and the approximate Bottom ')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("bottom_bottomapp.png")
    plt.savefig("bottom_bottomapp.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
def plot_directDisch_inverseDisch(values_q_pos_tg,q_history_tg,t_g,X):
    #plot Direct solver Discharge and inverse problm Discharge
    plt.figure(figsize=(8,4))          
    plt.plot(X,values_q_pos_tg,color='red',linestyle='none',marker='x',markersize=4., label=r'q reconstructed')
    plt.plot(X,q_history_tg,color='green',linestyle='none',marker='+',markersize=4., label=r'q from direct solver')    # if its a problem remove lm : size of the line graph
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'$t_f\;$={t_g} s ')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    plt.savefig("dirct_inverse_discharge.pdf",format="pdf", bbox_inches="tight" )
    #plt.savefig("dirct_inverse_discharge.png")
    plt.show()
    
    
    
def plot_given_time_inv_average(X,Batxj,B_T):
        
   ###Plot the bottom and the bottom appr
    plt.figure(figsize=(8,4))  
    plt.plot(X,Batxj,color='black',linestyle='-',lw=2.5, label=r'$b$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,B_T,color='red',linestyle='none',marker='o',markerfacecolor='none', label=r'$b\;\mathrm{ average}\; \mathrm{ reconstructed}$')
    #plt.title('Bottom and the average approximate Bottom ')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    plt.savefig("bottom_bottomappaverage.pdf",format="pdf", bbox_inches="tight" )
    #plt.savefig("bottom_bottomappaverage.png")
    plt.show()
    
    
    
    
    
##### Plots for Suf. conditions

def plotlamda1andlamda2(t_interval,min1l,max2l):
    # First figure: λ1
    plt.figure(figsize=(8, 4))
    plt.plot(t_interval,min1l,color='black',linestyle='none',marker='x',markersize=3., label=r'$\min_{I}\,\lambda_1$')
    plt.legend(loc='center right')
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel('t (s)')
    plt.show()


    # Second figure: λ2
    plt.figure(figsize=(8, 4))
    plt.plot(t_interval,max2l,color='black',linestyle='none',marker='x',markersize=3., label=r'$\max_{I}\,\lambda_2$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.savefig("lambda2.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
    

def plotFr(fr_at_t,X):
    plt.figure(figsize=(8, 4))
    plt.plot(X,fr_at_t,color='black',linestyle='none',marker='x',markersize=3., label=r'$Fr$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.xlabel('x (m)')
    plt.savefig("Fr.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
    
#### Plots for the case of noisy surface 
def plot_noisycase(X,Batxj,h_t_g,eta_noisy,t_g,eta_smooth,B_appx_inv):
    
        
    #plot exact and noisy
    plt.figure(figsize=(8,4))
    plt.plot(X,Batxj+h_t_g,color='blue',linestyle='-',marker='o',markerfacecolor='none', label=r'$\zeta$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,eta_noisy,color='red',linestyle='-',marker='*', label=r'Noisy $\zeta$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'$t_f\;$={t_g} s')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("exactand_Noisy_surfaces.png")
    plt.savefig("exactand_Noisy_surfaces.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
    #plot smoothed and exact eta 
    plt.figure(figsize=(8,4))
    plt.plot(X,Batxj+h_t_g,color='blue',linestyle='-',marker='o',markerfacecolor='none', label=r'$\zeta$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,eta_smooth,color='red',linestyle='-',marker='*', label=r'Smoothed noisy  $\zeta$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'$t_f\;$={t_g} s')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("exactand_Noisy_surfaces.png")
    plt.savefig("exactand_smoth_surfaces.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
    
    # plot exact surface bottom and bottom from noisy data   
    plt.figure(figsize=(8,4))
    plt.plot(X,Batxj+h_t_g,color='blue',linestyle='-',marker='x',markerfacecolor='none', label=r'$\zeta$')    # if its a problem remove lm : size of the line graph
    plt.plot(X,Batxj,color='black',linestyle='-',lw=2.5, label=r'$B$') 
    plt.plot(X,B_appx_inv,color='red',linestyle='none',marker='o', markerfacecolor='none',label=r'$B$ approximated from noisy $\zeta$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'$t_f\;$={t_g} s')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("exactand_Noisy_surfaces_bottomandbottomappr.png")
    plt.savefig("exact_surface_and_exactbottom_noisybottom.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
    
    #plot exact bottom and bottom from noisy data
    plt.figure(figsize=(8,4))
    plt.plot(X,Batxj,color='black',linestyle='-',lw=2.5, label=r'$B$') 
    plt.plot(X,B_appx_inv,color='red',linestyle='none',marker='o', markerfacecolor='none',label=r'$B$ approximated from noisy $\zeta$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    #plt.title(f'Surfaces at t={t_g} s')
    plt.grid(True)
    plt.legend(); plt.tight_layout()
    #plt.savefig("bottom_Noisy_bottom.png")
    plt.savefig("Noisy_bottom_and_exavt bottom.pdf",format="pdf", bbox_inches="tight" )
    plt.show()
    
         
