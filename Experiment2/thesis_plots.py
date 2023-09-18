import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle


plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{mathrsfs}",
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": "12",
    }
)

losslist = ["MeanPowerLoss2", "LPNormLoss2", "CrossEntropyLoss"]


def Plot_Function_Surface(show=False, save=True):
    """
    This function plots the output of the CrossEntropy Trained function
    """
    data = np.load("npfiles/CrossEntropyLoss_training.npz")
    Xfunc = data["Xfunc"]
    Yfunc = data["Yfunc"]
    Zfunc = data["Zfunc"]

    fig, ax = plt.subplots(figsize=(390 / 72, 390 / 72))

    plt.imshow(Zfunc, cmap=cm.magma)
    xticks = [0, len(Zfunc) / 2, len(Zfunc)]
    xticklabels = ["$0$", "$0.5$", "$1$"]
    plt.xticks(xticks, xticklabels)
    plt.yticks(xticks, xticklabels)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title("Network output for a trained network")
    if save:
        plt.savefig("plots/Network_output.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def Plot_Dataset(show=False, save=True):
    ## Import dataset
    with open("npfiles/dataset.npy", "rb") as file:
        dataset = pickle.load(file)

    inputs, targets = dataset["inputs"], dataset["targets"]

    xlist = [0, 1]
    plt.fill_between(xlist, y1=xlist, y2=1, color=cm.magma(0.8), alpha=0.2)
    plt.fill_between(xlist, y1=0, y2=xlist, color=cm.magma(0.2), alpha=0.2)
    label1, label2 = True, True

    for i in range(len(inputs)):
        if targets[i] == 1:
            plt.plot(inputs[i, 0], inputs[i, 1], "o", color=cm.magma(0.8))
            if label1:
                plt.plot(
                    inputs[i, 0],
                    inputs[i, 1],
                    "o",
                    color=cm.magma(0.8),
                    label="target $1$",
                )
                label1 = False
        else:
            plt.plot(inputs[i, 0], inputs[i, 1], "o", color=cm.magma(0.2))
            if label2:
                plt.plot(
                    inputs[i, 0],
                    inputs[i, 1],
                    "o",
                    color=cm.magma(0.2),
                    label="target $0$",
                )
                label2 = False

    plt.title("Dataset")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    if save:
        plt.savefig("plots/Dataset.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


Plot_Function_Surface()
Plot_Dataset()
