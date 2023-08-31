import jax.numpy as np
import jax
import numpy as onp
from rich.progress import track
import time


######################
#Create Dataset
N_data = 100
x, y = onp.random.rand(N_data), onp.random.rand(N_data)
c = np.array(y>=x).astype(int)
dataset = np.transpose(np.array([x,y,c]))
######################

#Helper functions:
def network(x,y,theta,a):
    return 1/(1+np.exp(-a*theta[0]*x-a*theta[1]*y))

def fisher_info_matrix(dataset, theta, a):
    dataset, theta = np.array(dataset),np.array(theta)
    N = len(dataset)

    def mapping_func(i):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        I11 = ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,0]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
        I22 = ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,1]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
        I12 = (2*(dataset[i,2]-n_out)*n_out**2)**2 * (
                a**2 *dataset[i,1]*dataset[i,0]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1]))
        return I11,I22,I12
    
    map = jax.vmap(mapping_func)
    i_list = np.arange(N)
    I11_list, I22_list, I12_list = map(i_list)
    I11, I22, I12 = np.mean(I11_list, axis=0),np.mean(I22_list, axis=0),np.mean(I12_list, axis=0) 
    return np.array([[I11,I12],[I12,I22]])


def fisher_info_matrix_alt(dataset, theta, a):
    N = len(dataset)
    I11, I12, I22 = 0,0,0
    for i in range(N):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        I11 += ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,0]*onp.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
        I22 += ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,1]*onp.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
        I12 += (2*(dataset[i,2]-n_out)*n_out**2)**2 * (
                a**2 *dataset[i,1]*dataset[i,0]*onp.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1]))
    return onp.array([[I11,I12],[I12,I22]])/N

def del_gij_del_xk(i,j,k,theta,g,a):
    e = 1e-4
    dktheta = np.where(np.arange(len(theta))==k, np.array(theta)+e, np.array(theta))
    return (fisher_info_matrix(dataset,dktheta,a)[i,j]-g[i,j])/e

def christoffel(i,j,k, theta, g, ig, a):
    def mapping_func(m,l):
        return 0.5*ig[i,m]*(del_gij_del_xk(m,k,l,theta,g,a) + 
                            del_gij_del_xk(m,l,k,theta,g,a) - 
                            del_gij_del_xk(k,l,m,theta,g,a))
    
    map = jax.vmap(mapping_func, in_axes = (0,0))
    i_list = np.arange(len(theta))
    christoffel_list = map(i_list,i_list)
    return np.sum(christoffel_list,axis=0)


#####################################
#Final function:
def Scalar_curvature(dataset,theta,a):
    g = fisher_info_matrix(dataset,theta,a)
    ig = np.linalg.inv(g)

    def mapping_func_full(mu,v,L,sigma,g,ig,dataset,theta,a):
        e = 1e-4
        dLtheta = np.where(np.arange(len(theta))==L, np.array(theta)+e, np.array(theta))
        dvtheta = np.where(np.arange(len(theta))==v, np.array(theta)+e, np.array(theta))
        c1 = (christoffel(L,mu,v,dLtheta,g,ig,a)-christoffel(L,mu,v,theta,g,ig,a))/e
        c2 = (christoffel(L,mu,L,dvtheta,g,ig,a)-christoffel(L,mu,L,theta,g,ig,a))/e
        c3 = christoffel(sigma,mu,v,theta,g,ig,a)*christoffel(L,L,sigma,theta,g,ig,a)
        c4 = christoffel(sigma,mu,L,theta,g,ig,a)*christoffel(L,v,sigma,theta,g,ig,a)
        return ig[mu,v]*(c1-c2+c3-c4)
    
    mapping_func = lambda mu,v,L,sigma: mapping_func_full(mu,v,L,sigma, g=g, ig=ig,dataset=dataset,theta=theta,a=a)
    map = jax.vmap(mapping_func, in_axes = (0,0,0,0))
    i_list = np.arange(len(theta))
    curvature_list = map(i_list,i_list,i_list,i_list)

    return np.sum(curvature_list,axis=0)
############################################


if __name__ == '__main__':
    sn = time.time()
    fisher_info_matrix(dataset,theta=[1,1],a=5)
    en = time.time()
    so = time.time()
    fisher_info_matrix_alt(dataset,theta=[1,1],a=5)
    eo = time.time()

    print(f"old duration: {eo-so}, new duration: {en-sn}")

    '''
    #######################################
    #Calculation of the surface
    theta1 = np.linspace(-2,1,100)
    theta2 = np.linspace(-1,2,100)
    X, Y = np.meshgrid(theta1, theta2)
    t_list = np.load("training.npz")['t_list']
    l_list = np.load("training.npz")['l_list']
    acc = np.load("training.npz")['acc']

    Z = onp.zeros_like(X)
    for i, theta1_ in enumerate(theta1):
        print(f"Calculating scalar curvatures done {i}%")
        for j in track(range(len(theta2))):
            Z[j,i] = Scalar_curvature(dataset=dataset, theta=[theta1_,theta2[i]], a = 5)

    Zpath = []
    for i in range(len(t_list[0])):
        print(f"Calculating curvature path done {100*i/len(t_list[0])}%")
        Zpath.append(Scalar_curvature(dataset, theta=[t_list[0][i],t_list[1][i]], a=5))

    np.savez("curvature_plot.npz", X=X,Y=Y,Z=Z,t_list=t_list,Zpath=Zpath, allow_pickle=True)
    '''