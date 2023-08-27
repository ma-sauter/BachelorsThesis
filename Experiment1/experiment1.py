import matplotlib.pyplot as plt 
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from rich.progress import track

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
        _loss += np.abs(dataset[i,2]-network(dataset[i,0],dataset[i,1],theta,a))
    return 1/N*_loss

def analytic_gradient(dataset, theta,a):
    N = len(dataset)
    grad1 = 0
    grad2 = 0
    for i in range(N):
        n_out = network(dataset[i,0],dataset[i,1],theta,a)
        grad1 += 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,0]*(1/n_out-1))
        grad2 += 2*(dataset[i,2]-n_out)*n_out**2 *(-a*dataset[i,1]*(1/n_out-1))
    return np.array(grad1,grad2)/N

def gradient(dataset, theta, a):
    epsilon = 1e-6
    grad1 = (loss(dataset, theta+[epsilon,0],a) - loss(dataset, theta-[epsilon,0],a))/2/epsilon
    grad2 = (loss(dataset, theta+[0,epsilon],a) - loss(dataset, theta-[0,epsilon],a))/2/epsilon
    return np.array(grad1,grad2)

def fisher_info_matrix(dataset, theta, a):
    N = len(dataset)
    I11, I12, I22 = 0,0,0
    for i in range(N):
        I11 += 0


def training(n_epochs, dataset, a, learning_rate):
    theta = np.array([np.random.rand(1)[0]*2-2,np.random.rand(1)[0]*2])
    theta_list = [theta]
    loss_list = [loss(dataset=dataset,theta=theta,a=a)]
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

t_list, l_list, acc = training(10000, dataset= dataset, a = 10, learning_rate=1e-2)
print(f"Accuracy: {acc[-1]}")
t_list = np.transpose(t_list)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

theta1 = np.linspace(-2,1,100)
theta2 = np.linspace(-1,2,100)
X, Y = np.meshgrid(theta1, theta2)
Z = np.zeros_like(X)
for i, theta1_ in enumerate(theta1):
    for j, theta2_ in enumerate(theta2):
        Z[j,i] = loss(dataset=dataset, theta=[theta1_,theta2_], a = 10)

surf = ax.plot_surface(X, Y, Z, cmap=cm.magma,
                       linewidth=0, antialiased=True)
path = ax.plot(t_list[0],t_list[1], l_list,color = 'mediumseagreen', zorder=10)

plt.show()
