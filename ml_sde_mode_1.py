# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:06:10 2019

@author: zirklej
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 08 12:15:09 2019

@author: zirklej

noisy stimulation case

mode 1 system

"""

from numpy import *
from random import gauss
from random import seed
import matplotlib.pyplot as plt
import time

def m_inf(v,v_m1,v_m2):
    return 1/(1+exp(-2*(v-v_m1)/v_m2))

def w_inf(v,v_w1,beta_w):
    return 1/(1+exp(-2*(v-v_w1)/beta_w))

def tau(v,epsilon,v_w1,beta_tau):
    return (2/epsilon)*(1/(exp((v-v_w1)/(2*beta_tau))+exp((v_w1-v)/(2*beta_tau))))

def H_inf(v,sigma_s):
    return 1/(1+exp(-(v/sigma_s)))


def parameter():
    
    # parameters
    gCa=1   # 0 - number in array
    gK=3.1    # 1  (should be 3.1 as stated in published article)
    gL=0.5  # 2
    vCa=1   # 3
    vK=-0.7 # 4
    vL=-0.4 # 5
    v_m1=-0.01  # 6
    v_m2=0.15   # 7
    v_w1=0.07 # 8
    beta_w=linspace(0.134,0.094,100) # 9
    beta_w=beta_w[90]
    beta_tau=linspace(0.061,0.081,100) # 10
    beta_tau=beta_tau[90]
    I=0.04 # 11
    epsilon1=0.03  # 12
    epsilon2=1.3*epsilon1  # 13
    epsilon3=array([0.5*epsilon1,(epsilon1+epsilon2)/2,1.5*epsilon1,2*epsilon1])
    epsilon3=epsilon3[l]  #14
    
    # synaptic parameters
    g_syn1=0.0006    # 15
    g_syn2=0.0006    # 16
    g_syn3=linspace(0,1,numfiles)
    g_syn3=g_syn3[k]  #17
    v_syn=0.5   # 18
    alpha_s=5   # 19
    beta_s=0.2  # 20
    theta_v=0.# 21
    sigma_s=0.2 # 22

    # bundle parameters
    params=array([gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
            alpha_s,beta_s,theta_v,sigma_s])
    
    return params

# define drift (f) and diffusion (g) functions where the SDE is
# dx=f(x,t)dt+g(x,t)dW

# define noise

def noise(dt,N,numfiles):
    # amplitude = 0 should recreate data for Figure 8 from Rubchinsky's paper
    # amplitude=linspace(0.0,0.02,num=40,endpoint=True)
    # amplitude=amplitude[39]
    # w=array([amplitude*gauss(0.,1.)*sqrt(dt) for i in range(N)])
    return zeros(N)

#define first neuron

def v1_drift(x):
    v1=x[0]
    w1=x[1]
    s2=x[5]
    s3=x[8]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    volt_drift=-gCa*m_inf(v1,v_m1,v_m2)*(v1-vCa)-gK*w1*(v1-vK)-gL*(v1-vL)-g_syn2*(v1-v_syn)*s2-g_syn3*(v1-v_syn)*s3+I
    return volt_drift

def w1_drift(x):
    v1=x[0]
    w1=x[1]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    w_drift=(w_inf(v1,v_w1,beta_w)-w1)/tau(v1,epsilon1,v_w1,beta_tau)
    return w_drift

def v1_diffusion(x):
    v1=x[0]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    diffus=-gK*(v1-vK)
    return diffus

def s1_drift(x):
    v1=x[0]
    s1=x[2]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    syn=alpha_s*(1-s1)*H_inf(v1-theta_v,sigma_s)-beta_s*s1
    return syn

# define second neuron

def v2_drift(x):
    s1=x[2]
    v2=x[3]
    w2=x[4]
    s3=x[8]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    volt_drift=-gCa*m_inf(v2,v_m1,v_m2)*(v2-vCa)-gK*w2*(v2-vK)-gL*(v2-vL)-g_syn1*(v2-v_syn)*s1-g_syn3*(v2-v_syn)*s3+I
    return volt_drift

def v2_diffusion(x):
    v2=x[3]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    diffus=-gK*(v2-vK)
    return diffus

def w2_drift(x):
    v2=x[3]
    w2=x[4]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    w_drift=(w_inf(v2,v_w1,beta_w)-w2)/tau(v2,epsilon2,v_w1,beta_tau)
    return w_drift

def s2_drift(x):
    v2=x[3]
    s2=x[5]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    syn=alpha_s*(1-s2)*H_inf(v2-theta_v,sigma_s)-beta_s*s2
    return syn

# define third neuron

def v3_drift(x):
    v3=x[6]
    w3=x[7]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    volt_drift=-gCa*m_inf(v3,v_m1,v_m2)*(v3-vCa)-gK*w3*(v3-vK)-gL*(v3-vL)+I
    return volt_drift

def w3_drift(x):
    v3=x[6]
    w3=x[7]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    w_drift=(w_inf(v3,v_w1,beta_w)-w3)/tau(v3,epsilon3,v_w1,beta_tau)
    return w_drift

def v3_diffusion(x):
    v3=x[6]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    diffus=-gK*(v3-vK)
    return diffus

def s3_drift(x):
    v3=x[6]
    s3=x[8]
    (gCa,gK,gL,vCa,vK,vL,v_m1,v_m2,v_w1,beta_w,beta_tau,I,epsilon1,epsilon2,epsilon3,g_syn1,g_syn2,g_syn3,v_syn,
     alpha_s,beta_s,theta_v,sigma_s)=parameter()
    syn=alpha_s*(1-s3)*H_inf(v3-theta_v,sigma_s)-beta_s*s3
    return syn

# time span
T=20000
dt=0.01
tspan=arange(0.,T,dt)
N=len(tspan) # number of time steps

# initial conditions
v10=0.1
w10=0.376
s10=0.86
v20=-0.29
w20=0.127
s20=0.64
v30=0.
w30=0.
s30=0.

# bundle ICs
x0=[v10,w10,s10,v20,w20,s20,v30,w30,s30]

numfiles=20 # number of grid points to vary gsyn,3 given a fixed epsilon3
for j in range(20,2*numfiles):  # 4 = number of epsilon3 values to cycle through
    print(j)
    start=time.time()
    
    # define global variables to use in parameter function 
    k=int(j%numfiles)   # varies g_syn,3, i.e. fast 
    l=int(floor(j/numfiles))    # varies epsilon3, i.e. slow
    
    # define numerical solution vector
    # x[0,] = v1
    # x[1,] = w1
    # x[2,] = s1
    # x[3,] = v2
    # x[4,] = w2
    # x[5,] = s2
    # x[6,] = v3
    # x[7,] = w3
    # x[8,] = s3
    x=zeros((9,N))   
    x[:,0]=x0

    v1_wiener=noise(dt,N,numfiles)
    v2_wiener=noise(dt,N,numfiles)
    v3_wiener=noise(dt,N,numfiles)
    
    # implement euler-maruyama scheme to solve SDE
    for i in range(1,N):
        
        # first neuron
        
        x[0,i]=x[0,i-1]+v1_drift(x[:,i-1])*dt+v1_diffusion(x[:,i-1])*v1_wiener[i]
        x[1,i]=x[1,i-1]+w1_drift(x[:,i-1])*dt
        if x[1,i]>1:
            x[1,i]=1
        if x[1,i]<0:
            x[1,i]=0
        x[2,i]=x[2,i-1]+s1_drift(x[:,i-1])*dt
        
        # second neuron 
        
        x[3,i]=x[3,i-1]+v2_drift(x[:,i-1])*dt+v2_diffusion(x[:,i-1])*v2_wiener[i]
        x[4,i]=x[4,i-1]+w2_drift(x[:,i-1])*dt
        if x[4,i]>1:
            x[4,i]=1
        if x[4,i]<0:
            x[4,i]=0
        x[5,i]=x[5,i-1]+s2_drift(x[:,i-1])*dt
        
        # third neuron 
        
        x[6,i]=x[6,i-1]+v3_drift(x[:,i-1])*dt+v3_diffusion(x[:,i-1])*v3_wiener[i]
        x[7,i]=x[7,i-1]+w3_drift(x[:,i-1])*dt
        if x[7,i]>1:
            x[7,i]=1
        if x[7,i]<0:
            x[7,i]=0
        x[8,i]=x[8,i-1]+s3_drift(x[:,i-1])*dt

    # write data to file
    fin=open("mode_1_epsilon3_"+str(l)+"_gsyn3_"+str(k)+".txt","w+")
    soln=column_stack((tspan,transpose(x))) # concatenate time and solution arrays
    savetxt(fin,soln,fmt='%f')
    fin.close() 
    end=time.time()
    print(end-start)
    