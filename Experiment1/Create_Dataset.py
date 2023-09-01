import numpy as np
import matplotlib.pyplot as plt

######################
#Create Dataset
N_data = 100
x, y = np.random.rand(N_data), np.random.rand(N_data)
c = np.array(y>=x).astype(int)
dataset = np.transpose(np.array([x,y,c]))

#Plot
colorlist = []
for i in range(len(dataset)):
    if dataset[i,2] ==1:
        plt.plot(dataset[i,0],dataset[i,1], "o",color = "green")
    else:
        plt.plot(dataset[i,0],dataset[i,1], "o",color = "red")
    
plt.show()
plt.close()

np.save("Dataset.npy", dataset, allow_pickle=True)
######################