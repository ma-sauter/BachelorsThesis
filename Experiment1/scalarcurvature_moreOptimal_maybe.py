from math import e
import matplotlib.pyplot as plt 
import numpy as np
import numpy
from tqdm import tqdm
from rich.progress import track
import jax
import time
from joblib import Parallel, delayed

######################
#Create Dataset
N_data = 100
x, y = numpy.random.rand(N_data), numpy.random.rand(N_data)
c = np.array(y>=x).astype(int)
dataset = np.transpose(np.array([x,y,c]))
######################


def network(x,y,theta,a):
    return 1/(1+np.exp(-a*theta[0]*x-a*theta[1]*y))

def fisher_info_matrix11(theta, a):
    N = len(dataset)
    I11 = 0
    for i in range(N):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        I11 += ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,0]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
    return I11/N
def fisher_info_matrix12(theta, a):
    N = len(dataset)
    I12 = 0
    for i in range(N):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        I12 += (2*(dataset[i,2]-n_out)*n_out**2)**2 * (
                a**2 *dataset[i,1]*dataset[i,0]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1]))
    return I12/N
def fisher_info_matrix22(theta, a):
    N = len(dataset)
    I22 = 0
    for i in range(N):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        I22 += ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,1]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
    return I22/N
fisher_info_matrix = np.array([[fisher_info_matrix11, fisher_info_matrix12],[fisher_info_matrix12,fisher_info_matrix22]])

def del_gij_del_xk(i,j,k,theta,a):
    e = 1e-4
    d1theta, d2theta = np.copy(theta), np.copy(theta)
    d1theta[k] = d1theta[k]+e
    d2theta[k] = d2theta[k]-e
    return (fisher_info_matrix[i,j](d1theta,a)-fisher_info_matrix[i,j](d2theta,a))/2/e

def christoffel(i,j,k,theta,ig,a):
    symbol = 0
    for m in range(len(theta)):
        for l in range(len(theta)):
            symbol += 0.5*ig[i,m]*(del_gij_del_xk(m,k,l,theta,a) + del_gij_del_xk(m,l,k,theta,a) - del_gij_del_xk(k,l,m,theta,a))
    return symbol

def Riemannian_curvature_tensor(i,j,k,l,theta,ig,a):
    e = 1e-4
    dktheta, dltheta = np.copy(theta), np.copy(theta)
    dktheta[k] = dktheta[k]+e
    dltheta[l] = dltheta[l]+e
    c1 = (christoffel(i,j,l,dktheta,ig,a)-christoffel(i,j,l,theta,ig,a))/e
    c2 = (christoffel(i,j,k,dltheta,ig,a)-christoffel(i,j,k,theta,ig,a))/e
    c34 = 0
    for m in range(len(theta)):
        c34 += christoffel(i,m,k,theta,ig,a)*christoffel(m,j,l,theta,ig,a)
        c34 -= christoffel(i,m,l,theta,ig,a)*christoffel(m,j,k,theta,ig,a)
    return c1-c2+c34

def Ricci_tensor(ij,theta,ig,a):
    i,j = ij
    tensor = 0
    for m in range(len(theta)):
        tensor += Riemannian_curvature_tensor(m,i,m,j,theta,ig,a)
    return tensor

def Scalar_curvature(theta,a):
    g = np.array([[fisher_info_matrix[0,0](theta,a), fisher_info_matrix[0,1](theta,a)],
                  [fisher_info_matrix[1,0](theta,a), fisher_info_matrix[1,1](theta,a)]])
    ig = np.linalg.inv(g)
    curv = 0

    par_list = []
    for i in range(len(theta)):
        for j in range(len(theta)):
            par_list.append(np.array([i,j]))
    
    def map_func(ij):
        return ig[ij[0],ij[1]]*Ricci_tensor(ij,theta,ig,a)
    
    listofvalues = Parallel(n_jobs=-1)(delayed(map_func)(liste) for liste in par_list)
    return np.sum(listofvalues)

Scalar_curvature([1,1],5)
old = time.time()
print(Scalar_curvature([1,1],5))
after = time.time()
print(f"This took {after-old}")