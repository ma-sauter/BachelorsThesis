from keras.datasets import mnist
import matplotlib.pyplot as plt 
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
n_plots = 4
fig, ax = plt.subplots(1,n_plots)
for i in range(n_plots):  
    axis = ax[i]
    axis.imshow(train_X[i], cmap=plt.get_cmap('bone'))
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    axis.set_xticks(np.arange(28)-0.5)
    axis.set_yticks(np.arange(28)-0.5)
    axis.tick_params(left = False, bottom = False)
    axis.grid(color='gray', linestyle='-', linewidth=0.07)

plt.savefig('mnist_plot.pdf', bbox_inches='tight')