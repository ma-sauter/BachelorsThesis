import numpy as np
import matplotlib.pyplot as plt
import pickle

######################
# Create Dataset
N_data = 100
x, y = np.random.rand(N_data), np.random.rand(N_data)
c = np.array(y >= x).astype(int)
dataset = {"inputs": np.transpose(np.array([x, y])), "targets": c}

# Plot
inputs = dataset["inputs"]
targets = dataset["targets"]
for i in range(inputs.shape[0]):
    if targets[i] == 1:
        plt.plot(inputs[i, 0], inputs[i, 1], "o", color="green")
    else:
        plt.plot(inputs[i, 0], inputs[i, 1], "o", color="red")

plt.show()
plt.close()

# Save dictionary
with open("npfiles/dataset.npy", "wb") as file:
    pickle.dump(dataset, file)
######################
