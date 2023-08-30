import matplotlib.pyplot as plt 
import jax.numpy as np
from tqdm import tqdm
from rich.progress import track
import jax

######################
#Create Dataset
N_data = 100
x, y = np.random.rand(N_data), np.random.rand(N_data)
c = np.array(y>=x).astype(int)
dataset = np.transpose(np.array([x,y,c]))

#Plot
if False:
    colorlist = []
    for i in range(len(dataset)):
        if dataset[i,2] ==1:
            plt.plot(dataset[i,0],dataset[i,1], "o",color = "green")
        else:
            plt.plot(dataset[i,0],dataset[i,1], "o",color = "red")
    
    plt.show()
    plt.close()
######################

def network(x,y,theta,a):
    return 1/(1+np.exp(-a*theta[0]*x-a*theta[1]*y))
def loss(dataset, theta,a):
    N = len(dataset)
    _loss = 0
    for i in range(N):
        _loss += (dataset[i,2]-network(dataset[i,0],dataset[i,1],theta,a))**2
    return 1/N*_loss

def gradient(dataset, theta,a):
    N = len(dataset)
    grad1 = 0
    grad2 = 0
    for i in range(N):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        grad1 += 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,0]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1]))
        grad2 += 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,1]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1]))
    return np.array([grad1,grad2])/N

def numerical_gradient(dataset, theta, a):
    epsilon = 1e-6
    grad1 = (loss(dataset, theta+[epsilon,0],a) - loss(dataset, theta-[epsilon,0],a))/2/epsilon
    grad2 = (loss(dataset, theta+[0,epsilon],a) - loss(dataset, theta-[0,epsilon],a))/2/epsilon
    return np.array([grad1,grad2])

@jax.jit
def fisher_info_matrix(dataset, theta, a):
    N = len(dataset)
    I11, I12, I22 = 0,0,0
    for i in range(N):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        I11 += ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,0]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
        I22 += ( 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,1]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1])) )**2
        I12 += (2*(dataset[i,2]-n_out)*n_out**2)**2 * (
                a**2 *dataset[i,1]*dataset[i,0]*np.exp(-a*theta[0]*dataset[i,0] -a*theta[1]*dataset[i,1]))
    return np.array([[I11,I12],[I12,I22]])

def Scalar_curvature(dataset, theta, a):
    global g
    global ig
    g = fisher_info_matrix(dataset, theta, a)
    ig = np.linalg.inv(g)

    def christoffel(i,j,k, theta):
        def del_gij_del_xk(i,j,k,theta):
            e = 1e-4
            dtheta = np.copy(theta)
            dtheta[l] = dtheta[l]+e
            return (fisher_info_matrix(dataset,dtheta,a)[i,j]-g[i,j])/e
        symbol = 0
        for m in range(len(theta)):
            for l in range(len(theta)):
                symbol += 0.5*ig[i,m]*(del_gij_del_xk(m,k,l,theta) + del_gij_del_xk(m,l,k,theta) + del_gij_del_xk(k,l,m,theta))
        return symbol
    
    par_index_list = []
    for mu in range(len(theta)):
        for v in range(len(theta)):
            for L in range(len(theta)):
                for sigma in range(len(theta)):
                    par_index_list.append(np.array([mu,v,L,sigma]))
    
    def vmap_func(muvLsList):
        mu, v, L, s = muvLsList
        e = 1e-4
        dLtheta, dvtheta = np.copy(theta), np.copy(theta)
        dLtheta[L] = dLtheta[L]+e
        dvtheta[v] = dvtheta[v]+e
        c1 = (christoffel(L,mu,v,dLtheta)-christoffel(L,mu,v,theta))/e
        c2 = (christoffel(L,mu,L,dvtheta)-christoffel(L,mu,L,theta))/e
        c3 = christoffel(s,mu,v,theta)*christoffel(L,L,s,theta)
        c4 = christoffel(s,mu,L,theta)*christoffel(L,v,s,theta)
        return ig[mu,v]*(c1-c2+c3-c4)
    map = jax.vmap(vmap_func)
    return np.sum(map(par_index_list))

    
    
    




##################################################
def training(n_epochs, dataset, a, learning_rate):
    theta = np.array([0.5,0.5])
    theta_list = [theta]
    loss_list = [loss(dataset,theta,a)]
    accuracy = []
    for i in track(range(n_epochs)):
        theta = theta - learning_rate*gradient(dataset=dataset,theta=theta,a=a)
        loss_list.append(loss(dataset=dataset,theta=theta,a=a))
        theta_list.append(theta)
        wrong_guesses = 0
        N = len(dataset)
        for i in range(N):
            wrong_guesses += (np.round(network(dataset[i,0],dataset[i,1],theta,a)) - dataset[i,2])**2
        accuracy.append((N-wrong_guesses)/N)

    return theta_list, loss_list, accuracy

#####################################################################################
#Training
t_list, l_list, acc = training(10000, dataset= dataset, a = 5, learning_rate=5e-3)
print(f"Accuracy: {acc[-1]}")
t_list = np.transpose(t_list)
np.save("training.npy", [t_list,l_list,acc], allow_pickle = True)
######################################################################################

######################################################################################
#Loss surface plot
theta1 = np.linspace(-2,1,100)
theta2 = np.linspace(-1,2,100)
X, Y = np.meshgrid(theta1, theta2)

Z = np.zeros_like(X)
for i, theta1_ in enumerate(theta1):
    for j, theta2_ in enumerate(theta2):
        Z[j,i] = loss(dataset=dataset, theta=[theta1_,theta2_], a = 5)
np.save("loss_surf_plot.npy", [X,Y,Z], allow_pickle=True)
######################################################################################

######################################################################################
#Fisher surface plot
def fisher_surf(t1,t2):
    t_list, l_list, acc = np.load("training.npy")
    theta1 = np.linspace(-2,2,100)
    theta2 = np.linspace(-2,2,100)
    X, Y = np.meshgrid(theta1, theta2)


    Z = np.zeros_like(X)
    for i, theta1_ in enumerate(theta1):
        print(f"{i}%")
        for j, theta2_ in enumerate(theta2):
            fisher_info11 = fisher_info_matrix(dataset=dataset, theta=[theta1_,theta2_], a = 5)[t1,t2]
            Z[j,i] = fisher_info11

    pathZ = []
    for i in range(len(t_list[0])):
        pathZ.append(fisher_info_matrix(dataset,theta=[t_list[0][i],t_list[1][i]], a = 5)[t1,t2])
    return [X,Y,Z,t_list,pathZ]
np.save("Fisher_surf_plot11", fisher_surf(0,0), allow_pickle=True)
np.save("Fisher_surf_plot12", fisher_surf(1,0), allow_pickle=True)
np.save("Fisher_surf_plot22", fisher_surf(1,1), allow_pickle=True)

######################################################################################

'''
######################################################################################
#Scalar Curvature

theta1 = np.linspace(-2,1,100)
theta2 = np.linspace(-1,2,100)
X, Y = np.meshgrid(theta1, theta2)

Z = np.zeros_like(X)
for i, theta1_ in enumerate(theta1):
    for j, theta2_ in enumerate(theta2):
        Z[j,i] = Scalar_curvature(dataset=dataset, theta=[theta1_,theta2_], a = 5)
np.save("curvature_plot.npy", [X,Y,Z], allow_pickle=True)
######################################################################################
'''



