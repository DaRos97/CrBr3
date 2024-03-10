import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
import functions as fs
import os,scipy
from pathlib import Path
import matplotlib.pyplot as plt
s_ = 20
import sys
#Here we rescale the DFT data to fix the orthogonal fields

#Pars
machine = fs.get_machine(os.getcwd())

h_ort_M = 0.55
h_ort_AA = 0.2

#1 - Import I
I = fs.get_dft_data(machine)
A_M = (fs.a1,fs.a2)

def comp_prop(Phi): #Computes interpolation of phi
    pts = Phi.shape[0]
    big_Phi = fs.extend(Phi,5)
    S_array = np.linspace(-2,3,5*pts,endpoint=False)
    fun_Phi = RBS(S_array,S_array,big_Phi)
    return fun_Phi

#2 - Interpolate I
fun = comp_prop(I)
new_I = fun(np.linspace(0,1,200,endpoint=False),np.linspace(0,1,200,endpoint=False))
J_M = float(fun(1/3,0))
J_AA = float(fun(0,0))
J_AA_expected = J_M*h_ort_AA/h_ort_M

if 0: #Plot initial interlayer
    print('Initial J_M: ',J_M,'\nInitial J_AA: ',J_AA,'\nExpected J_AA: ',J_AA_expected)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
    s = ax1.imshow(I,origin='lower')
    plt.colorbar(s)
    ax1.set_xlabel(r'$a_1$',size=s_)
    ax1.set_ylabel(r'$a_2$',size=s_)
    ax1.set_title("Raw DFT data",fontsize=s_+10)
    #
    s = ax2.imshow(new_I,origin='lower')
    plt.colorbar(s)
    ax2.set_xlabel(r'$a_1$',size=s_)
    ax2.set_ylabel(r'$a_2$',size=s_)
    ax2.set_title("Interpolated DFT data",fontsize=s_+10)
    plt.show()
    exit()

#3 - Fourier transform
fft = scipy.fft.fft2(new_I)
M,N = fft.shape
pp = M*N
if 0:   #Plot fft
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
    fft = np.roll(np.roll(fft,5,axis=0),5,axis=1)/pp
    s = ax1.imshow(np.real(fft),origin='lower')
    ax1.set_title('fft real part',fontsize=s_)
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    plt.colorbar(s)
    #
    s = ax2.imshow(np.imag(fft),origin='lower')
    ax2.set_title('fft imaginary part',fontsize=s_)
    ax2.set_xlim(0,10)
    ax2.set_ylim(0,10)
    plt.colorbar(s)
    plt.show()
    exit()
#4 - Find max and min coordinates of I (real space)
pts = new_I.shape[1]
iM = np.argmax(new_I)
iU = (iM//pts,iM%pts)
JU = new_I[iU[0],iU[1]]*pp
im = np.argmin(new_I)
iV = (im//pts,im%pts)
JV = new_I[iV[0],iV[1]]*pp
#5 - Define variables for the equation for Beta
J0 = np.real(fft[0,0])
J_M = J_M*pp
J_AA_expected = J_AA_expected*pp
def gamma(a,b):
    return 2*(np.cos(2*np.pi*a)+np.cos(2*np.pi*b)+np.cos(2*np.pi*(a-b)))
def delta(a,b):
    return 2*(np.cos(4*np.pi*a)+np.cos(4*np.pi*b)+np.cos(4*np.pi*(a-b)))
def epsilon(a,b):
    return 2*(np.cos(2*np.pi*(a+b))+np.cos(2*np.pi*(a-2*b))+np.cos(2*np.pi*(2*a-b)))
def kappa(a,b):
    return 2*(np.sin(2*np.pi*(2*a-b))-np.sin(2*np.pi*(a+b))-np.sin(2*np.pi*(a-2*b)))
gu = gamma(iU[0]/pts,iU[1]/pts)
gv = gamma(iV[0]/pts,iV[1]/pts)
du = delta(iU[0]/pts,iU[1]/pts)
dv = delta(iV[0]/pts,iV[1]/pts)
eu = epsilon(iU[0]/pts,iU[1]/pts)
ev = epsilon(iV[0]/pts,iV[1]/pts)
ku = kappa(iU[0]/pts,iU[1]/pts)
kv = kappa(iV[0]/pts,iV[1]/pts)
#6 - Solve equation
X = np.linspace(0,np.pi,198)
pu = eu*np.cos(X)+ku*np.sin(X)
pv = ev*np.cos(X)+kv*np.sin(X)
den = dv-gv/gu*du
num = gv/gu*pu-pv
equ_B = (np.cos(X) + num/den*(1 - du/gu) - pu/gu) * (J0-J_M) - ((J_AA_expected-J0)/6 + (JV-J0-gv/gu*(JU-J0))/den*(du/gu - 1) - (JU-J0)/gu)*3*(np.cos(X)+np.sqrt(3)*np.sin(X))
iB = np.argmin(np.absolute(equ_B))
if 0: #plot solution
    plt.figure()
    plt.plot(X,equ_B)
    plt.scatter(X[iB],equ_B[iB])
    plt.show()
    exit()
#7 - Compute the new Js
pu_ = pu[iB]
pv_ = pv[iB]
num_ = num[iB]
beta = X[iB]
alpha = (J0-J_M)/3/(np.cos(beta)+np.sqrt(3)*np.sin(beta))
J2 = alpha*np.exp(1j*beta)
J3 = (JV-J0-gv/gu*(JU-J0)+alpha*num_)/den
J1 = (JU-J0-du*J3-alpha*pu_)/gu

#8 - Construct new fft
#new_fft = np.copy(fft)
new_fft = np.zeros((pts,pts),dtype=complex)
new_fft[0,0] = fft[0,0]
new_fft[1,0] = new_fft[1,-1] = new_fft[0,-1] = new_fft[-1,0] = new_fft[-1,1] = new_fft[0,1] = J1 
new_fft[1,1] = new_fft[1,-2] = new_fft[-2,1] = J2
new_fft[2,-1] = new_fft[-1,-1] = new_fft[-1,2] = np.conjugate(J2)
new_fft[2,0] = new_fft[-2,0] = new_fft[2,-2] = new_fft[-2,2] = new_fft[0,-2] = new_fft[0,2] = J3
if 0: #plot new fft
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
    new_fft = np.roll(np.roll(new_fft,5,axis=0),5,axis=1)/pp
    s = ax1.imshow(np.real(new_fft),origin='lower')
    ax1.set_title('fft real part',fontsize=s_)
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    plt.colorbar(s)
    #
    s = ax2.imshow(np.imag(new_fft),origin='lower')
    ax2.set_title('fft imaginary part',fontsize=s_)
    ax2.set_xlim(0,10)
    ax2.set_ylim(0,10)
    plt.colorbar(s)
    plt.show()
    exit()

#9 - construct new interlayer coupling values
new_Phi = np.real(scipy.fft.ifft2(new_fft))
new_fun = comp_prop(new_Phi)
if 0:   #Plot old and new interlayer
    print("Original Phi:")
    print("Coinstant part: ",new_I.sum()/pts**2)
    print('M: ',float(fun(1/3,0)),'\nAA : ',float(fun(0,0)),'\n')
    print("New Phi:")
    print("Coinstant part: ",new_Phi.sum()/pts**2)
    print('M: ',float(new_fun(1/3,0)),'\nAA : ',float(new_fun(0,0)),'\n')
    print("Expected J_AA: ",float(fun(1/3,0))*h_ort_AA/h_ort_M)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
    s = ax1.imshow(new_I,origin='lower')
    plt.colorbar(s)
    ax1.set_xlabel(r'$a_1$',size=s_)
    ax1.set_ylabel(r'$a_2$',size=s_)
    ax1.set_title("original DFT data",fontsize=s_)
    #
    s = ax2.imshow(new_Phi,origin='lower')
    plt.colorbar(s)
    ax2.set_xlabel(r'$a_1$',size=s_)
    ax2.set_ylabel(r'$a_2$',size=s_)
    ax2.set_title("New data",fontsize=s_)
    plt.show()
    exit()

#10 - Save it
np.save('Data/CrBr3_interlayer_rescaled.npy',new_Phi)








