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


t_list, l_list, acc = training(10000, dataset= dataset, a = 5, learning_rate=5e-3)
print(f"Accuracy: {acc[-1]}")
t_list = np.transpose(t_list)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

theta1 = np.linspace(-2,1,100)
theta2 = np.linspace(-1,2,100)
X, Y = np.meshgrid(theta1, theta2)

Z = np.zeros_like(X)
for i, theta1_ in enumerate(theta1):
    for j, theta2_ in enumerate(theta2):
        Z[j,i] = loss(dataset=dataset, theta=[theta1_,theta2_], a = 5)

surf = ax.plot_surface(X, Y, Z, cmap=cm.magma,
                       linewidth=0, antialiased=True)
path = ax.plot(t_list[0],t_list[1], l_list, "-",color = 'mediumseagreen', zorder=10)

plt.title("loss surface and training evolution")
plt.show()
plt.close()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

theta1 = np.linspace(-2,2,100)
theta2 = np.linspace(-2,2,100)
X, Y = np.meshgrid(theta1, theta2)


Z = np.zeros_like(X)
for i, theta1_ in enumerate(theta1):
    print(f"{i}%")
    for j, theta2_ in enumerate(theta2):
        fisher_info11 = fisher_info_matrix(dataset=dataset, theta=[theta1_,theta2_], a = 5)[0,0]
        Z[j,i] = fisher_info11
surf = ax.plot_surface(X, Y, Z, cmap=cm.magma,
                       linewidth=0, antialiased=True)
ax.view_init(elev=90., azim=0.)

plt.title("F11 surface")
plt.show()
plt.close()