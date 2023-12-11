import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import seaborn as sns
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
            if label1 and label2 == False:
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


def Plot_Loss_Surfaces(show=False, save=True):
    fig, ax = plt.subplots(1, 3, figsize=(390 / 72, 390 / 72 / 3))
    fig.subplots_adjust(wspace=1)
    for i, lossname in enumerate(losslist):
        data = np.load(f"npfiles/{lossname}_training.npz")

        X = data["SurfX"]
        Y = data["SurfY"]
        Z = data["SurfZ"]
        t_list = data["t_list"]
        t_list0 = t_list[0][~np.isnan(t_list[0])]
        t_list1 = t_list[1][~np.isnan(t_list[1])]
        l_list = data["l_list"]
        acc = data["acc_list"]

        im = ax[i].imshow(
            Z,
            cmap=cm.magma,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
        )
        ax[i].invert_yaxis()
        ax[i].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
        if lossname[-1] != "LPNormLoss":
            ax[i].annotate(
                "",
                xy=(t_list0[-1], t_list1[-1]),
                xytext=(t_list0[-52], t_list1[-52]),
                arrowprops=dict(
                    arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0
                ),
            )
        ax[i].set_xlabel(r"$\theta_1$")

        plt.colorbar(im, ax=ax[i], fraction=0.08, pad=0.05, shrink=0.5)
        if lossname == "LPNormLoss2":
            ax[i].set_title("$L_2$-norm")
        if lossname == "CrossEntropyLoss":
            ax[i].set_title("Cross-entropy")
        if lossname == "MeanPowerLoss2":
            ax[i].set_title("Mean power of $n=2$")
    ax[0].set_ylabel(r"$\theta_2$")

    if save:
        plt.savefig("plots/LossSurfaces.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def Plot_Trace_Surfaces(lossname, show=False, save=True):
    curv_data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    fisher_data = np.load(f"npfiles/{lossname}_fisher_infos.npz")
    NTK_data = np.load(f"npfiles/{lossname}_ntk.npz")

    fig, ax = plt.subplots(2, 3, figsize=(390 / 72, 390 / 72 / 1.5))
    plt.subplots_adjust(wspace=0.7, hspace=0.4)

    #######################
    X, Y, Z11, Z12, Z22, t_list = (
        fisher_data["X"],
        fisher_data["Y"],
        fisher_data["Z11"],
        fisher_data["Z12"],
        fisher_data["Z22"],
        fisher_data["t_list"],
    )
    t_list0 = t_list[0][~np.isnan(t_list[0])]
    t_list1 = t_list[1][~np.isnan(t_list[1])]

    im = ax[0, 0].imshow(
        Z11,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        vmin=0,
        vmax=0.65,
    )
    """ ax[0, 0].contour(
        X,
        Y,
        Z11,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
        alpha=0.5,
    ) """
    ax[0, 0].invert_yaxis()
    ax[0, 0].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[0, 0].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[0, 0].set_title("$I_{11}$")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.08, pad=0.05, shrink=0.8)

    im = ax[0, 1].imshow(
        Z12,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        vmin=0,
        vmax=0.65,
    )
    """ ax[0, 1].contour(
        X,
        Y,
        Z12,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
        alpha=0.5,
    ) """
    ax[0, 1].invert_yaxis()
    ax[0, 1].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[0, 1].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[0, 1].set_title("$I_{12}$")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.08, pad=0.05, shrink=0.8)

    im = ax[0, 2].imshow(
        Z22,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        vmin=0,
        vmax=0.65,
    )
    """ ax[0, 2].contour(
        X,
        Y,
        Z22,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
        alpha=0.5,
    ) """
    ax[0, 2].invert_yaxis()
    ax[0, 2].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[0, 2].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[0, 2].set_title("$I_{22}$")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.08, pad=0.05, shrink=0.8)

    #########################################
    # Fisher traces
    im = ax[1, 0].imshow(
        Z11 + Z22,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
    )
    """ ax[1, 0].contour(
        X,
        Y,
        Z11 + Z22,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
        alpha=0.5,
    ) """
    ax[1, 0].invert_yaxis()
    ax[1, 0].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[1, 0].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[1, 0].set_title(r"$\mathrm{tr}(I)$")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.08, pad=0.05, shrink=0.8)

    #########################################
    # NTK Trace
    X, Y, Z = NTK_data["X"], NTK_data["Y"], NTK_data["Z"]
    im = ax[1, 1].imshow(
        Z,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
    )
    """ ax[1, 1].contour(
        X,
        Y,
        Z,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
        alpha=0.5,
    ) """
    ax[1, 1].invert_yaxis()
    ax[1, 1].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[1, 1].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[1, 1].set_title(r"$\mathrm{tr}(\Lambda)$")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.08, pad=0.05, shrink=0.8)

    #########################################
    # Curvature
    Z = curv_data["Z"]
    if lossname == losslist[0]:
        im = ax[1, 2].imshow(
            Z,
            cmap=cm.magma,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            vmax=200,
        )
    else:
        im = ax[1, 2].imshow(
            Z,
            cmap=cm.magma,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
        )
    if lossname == losslist[0]:
        clevels = [-300, -150, -50]
    if lossname == losslist[1]:
        clevels = [-300, -200, -100, -50, -25]
    if lossname == losslist[2]:
        clevels = [-10, -5, -2]
    """ ax[1, 2].contour(
        X,
        Y,
        Z,
        levels=clevels,
        linewidths=0.3,
        cmap=cm.magma_r,
        alpha=0.5,
    ) """
    ax[1, 2].invert_yaxis()
    ax[1, 2].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[1, 2].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[1, 2].set_title(r"$R$")
    plt.colorbar(im, ax=ax[1, 2], fraction=0.08, pad=0.05, shrink=0.8)

    ax[1, 0].set_xlabel(r"$\theta_1$")
    ax[1, 1].set_xlabel(r"$\theta_1$")
    ax[1, 2].set_xlabel(r"$\theta_1$")

    ax[0, 0].set_ylabel(r"$\theta_2$")
    ax[1, 0].set_ylabel(r"$\theta_2$")

    if save:
        plt.savefig(f"plots/{lossname}_tracecomparison.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def Plot_Curves(lossname, show=False, save=True):
    curv_data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    fisher_data = np.load(f"npfiles/{lossname}_fisher_infos.npz")
    NTK_data = np.load(f"npfiles/{lossname}_ntk.npz")

    fig, ax = plt.subplots(2, 1, figsize=(390 / 72, 390 / 72 / 1.5), sharex=True)
    ax[0].set_xticks([])
    plt.subplots_adjust(hspace=0)
    sparse_x = np.arange(len(curv_data["Zpath"])) * 20

    color1, color2 = cm.magma(0.2), cm.magma(0.7)
    ln1 = ax[0].plot(
        fisher_data["Zpath11"] + fisher_data["Zpath22"],
        color=color1,
        label=r"$\mathrm{tr}(I)$",
    )
    ax[0].tick_params("y", colors=color1)
    ax[0].yaxis.label.set_color(color1)
    ax[0].yaxis.set_label_position("left")
    ax[0].set_ylabel("Fisher Trace", color=color1)

    ax1 = ax[0].twinx()
    ln2 = ax1.plot(
        NTK_data["NTK_path"], "--", color=color2, label=r"$\mathrm{tr}(\Lambda)$"
    )
    ax1.yaxis.label.set_color(color2)
    ax1.tick_params("y", colors=color2)
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel("NTK Trace", color=color2)

    ax[1].plot(
        sparse_x, curv_data["Zpath"], "o", markersize=3.5, color="#00E88F", label=r"$R$"
    )

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    ax[1].legend()
    ax[1].set_xticks([0, 200, 400, 600, 800, 1000])
    ax[1].set_yticks([0, -100, -200, -300])
    ax[1].set_ylabel("scalar curvature")

    ax[1].set_xlabel("Epochs")
    ax[1].tick_params(labelright=True)

    if lossname == losslist[0]:
        ax[0].set_title(r"Traces and curvature for Mean Power loss of $n=2$")
    if lossname == losslist[1]:
        ax[0].set_title(r"Traces and curvature for $L_2$-norm loss")
    if lossname == losslist[2]:
        ax[0].set_title(r"Traces and curvature for Cross-entropy loss")

    # Setting 0 to be the same for upper and lower plot
    ax[1].tick_params(labelright=True)
    if lossname == losslist[0]:
        ax[1].set_ylim(-700, 0)
        ax[1].set_yticks([0, -200, -400, -600])
        ax1.set_yticks([20, 40, 60, 80])
        ax1.set_ylim(
            0,
        )
        ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax[0].set_ylim(
            0,
        )
    if lossname == losslist[1]:
        ax[1].set_ylim(-150, 0)
        ax[1].set_yticks([0, -50, -100, -150])
        ax1.set_yticks([20, 40])
        ax1.set_ylim(
            0,
        )
        ax[0].set_yticks([0.2, 0.4])
        ax[0].set_ylim(
            0,
        )
    if lossname == losslist[2]:
        ax[1].set_ylim(-80, 0)
        ax[1].set_yticks([0, -20, -40, -60, -80])
        ax1.set_yticks([25, 50, 75])
        ax1.set_ylim(
            0,
        )
        ax[0].set_yticks([2, 4, 6])
        ax[0].set_ylim(
            0,
        )

    # Setting the space at the edges to 0
    if lossname == losslist[0] or losslist[2]:
        ax[0].set_xlim(-10, 1010)
        ax[1].set_xlim(-10, 1010)
    if lossname == losslist[1]:
        ax[0].set_xlim(-2.8, 282.8)
        ax[1].set_xlim(-2.8, 282.8)
        ax[1].set_xticks([0, 50, 100, 150, 200, 250])

    if save:
        plt.savefig(f"plots/{lossname}_Curves.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def Plot_Loss_Surfaces(show=False, save=True):
    fig, ax = plt.subplots(3, 1, figsize=(390 / 72, 390 / 72 * 1.5))
    fig.subplots_adjust(hspace=0.4)
    for i, lossname in enumerate(losslist):
        data = np.load(f"npfiles/{lossname}_training.npz")

        X = data["SurfX"]
        Y = data["SurfY"]
        Z = data["SurfZ"]
        t_list = data["t_list"]
        t_list0 = t_list[0][~np.isnan(t_list[0])]
        t_list1 = t_list[1][~np.isnan(t_list[1])]
        l_list = data["l_list"]
        acc = data["acc_list"]

        im = ax[i].imshow(
            Z,
            cmap=cm.magma,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
        )
        cont = ax[i].contour(X, Y, Z, linewidths=0.3, cmap=cm.magma_r)
        ax[i].invert_yaxis()
        ax[i].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
        if lossname[-1] != "LPNormLoss":
            ax[i].annotate(
                "",
                xy=(t_list0[-1], t_list1[-1]),
                xytext=(t_list0[-52], t_list1[-52]),
                arrowprops=dict(
                    arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0
                ),
            )

        plt.colorbar(im, ax=ax[i], fraction=0.08, pad=0.05, shrink=0.7)
        if lossname == "LPNormLoss2":
            ax[i].set_title("(B)")
        if lossname == "CrossEntropyLoss":
            ax[i].set_title("(C)")
        if lossname == "MeanPowerLoss2":
            ax[i].set_title("(A)")
    ax[1].set_ylabel(r"$\theta_2$")
    ax[-1].set_xlabel(r"$\theta_1$")

    if save:
        plt.savefig("plots/LossSurfaces.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def Plot_Trace_Surfaces_Big(lossname, show=False, save=True):
    curv_data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    fisher_data = np.load(f"npfiles/{lossname}_fisher_infos.npz")
    NTK_data = np.load(f"npfiles/{lossname}_ntk.npz")

    fig, ax = plt.subplots(3, 2, figsize=(390 / 72, 390 / 72 * 1.3))
    ax = np.transpose(ax)
    # plt.subplots_adjust(wspace=0.7, hspace=0.3)

    #######################
    X, Y, Z11, Z12, Z22, t_list = (
        fisher_data["X"],
        fisher_data["Y"],
        fisher_data["Z11"],
        fisher_data["Z12"],
        fisher_data["Z22"],
        fisher_data["t_list"],
    )
    t_list0 = t_list[0][~np.isnan(t_list[0])]
    t_list1 = t_list[1][~np.isnan(t_list[1])]

    im = ax[0, 0].imshow(
        Z11,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        vmin=0,
        vmax=0.65,
    )
    ax[0, 0].contour(X, Y, Z11, levels=6, linewidths=0.3, cmap=cm.magma_r)
    ax[0, 0].invert_yaxis()
    ax[0, 0].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[0, 0].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[0, 0].set_title("$I_{11}$")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.08, pad=0.05, shrink=0.8)

    im = ax[0, 1].imshow(
        Z12,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        vmin=0,
        vmax=0.65,
    )
    ax[0, 1].contour(
        X,
        Y,
        Z12,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
    )
    ax[0, 1].invert_yaxis()
    ax[0, 1].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[0, 1].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[0, 1].set_title("$I_{12}$")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.08, pad=0.05, shrink=0.8)

    im = ax[0, 2].imshow(
        Z22,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        vmin=0,
        vmax=0.65,
    )
    ax[0, 2].contour(
        X,
        Y,
        Z22,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
    )
    ax[0, 2].invert_yaxis()
    ax[0, 2].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[0, 2].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[0, 2].set_title("$I_{22}$")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.08, pad=0.05, shrink=0.8)

    #########################################
    # Fisher traces
    im = ax[1, 0].imshow(
        Z11 + Z22,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
    )
    ax[1, 0].contour(
        X,
        Y,
        Z11 + Z22,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
    )
    ax[1, 0].invert_yaxis()
    ax[1, 0].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[1, 0].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[1, 0].set_title(r"$\mathrm{tr}(I)$")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.08, pad=0.05, shrink=0.8)

    #########################################
    # NTK Trace
    X, Y, Z = NTK_data["X"], NTK_data["Y"], NTK_data["Z"]
    im = ax[1, 1].imshow(
        Z,
        cmap=cm.magma,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
    )
    ax[1, 1].contour(
        X,
        Y,
        Z,
        levels=6,
        linewidths=0.3,
        cmap=cm.magma_r,
    )
    ax[1, 1].invert_yaxis()
    ax[1, 1].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[1, 1].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[1, 1].set_title(r"$\mathrm{tr}(\Lambda)$")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.08, pad=0.05, shrink=0.8)

    #########################################
    # Curvature
    X, Y, Z = curv_data["X"], curv_data["Y"], curv_data["Z"]
    if lossname == losslist[0]:
        im = ax[1, 2].imshow(
            Z,
            cmap=cm.magma,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            vmax=200,
        )
    else:
        im = ax[1, 2].imshow(
            Z,
            cmap=cm.magma,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
        )
    if lossname == losslist[0]:
        clevels = [-300, -150, -50]
    if lossname == losslist[1]:
        clevels = [-300, -200, -100, -50, -25]
    if lossname == losslist[2]:
        clevels = [-10, -5, -2]
    ax[1, 2].contour(
        X,
        Y,
        Z,
        levels=clevels,
        linewidths=0.3,
        cmap=cm.magma_r,
    )
    ax[1, 2].invert_yaxis()
    ax[1, 2].plot(t_list0[:-50], t_list1[:-50], "--", color="#00E88F")
    ax[1, 2].annotate(
        "",
        xy=(t_list0[-1], t_list1[-1]),
        xytext=(t_list0[-52], t_list1[-52]),
        arrowprops=dict(arrowstyle="-|>", color="#00E88F", shrinkA=0, shrinkB=0),
    )
    ax[1, 2].set_title(r"$R$")
    plt.colorbar(im, ax=ax[1, 2], fraction=0.08, pad=0.05, shrink=0.8)

    ax[0, 0].set_ylabel(r"$\theta_2$")
    ax[0, 1].set_ylabel(r"$\theta_2$")
    ax[0, 2].set_ylabel(r"$\theta_2$")

    ax[0, 2].set_xlabel(r"$\theta_1$")
    ax[1, 2].set_xlabel(r"$\theta_1$")

    fig.tight_layout()
    if save:
        plt.savefig(
            f"plots/{lossname}_tracecomparison_Big.pdf"
        )  # , bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# Plot_Function_Surface()
# Plot_Dataset()
# Plot_Loss_Surfaces(show=True)
# Plot_Trace_Surfaces("MeanPowerLoss2")
# Plot_Trace_Surfaces("LPNormLoss2")
# Plot_Trace_Surfaces("CrossEntropyLoss")
# Plot_Trace_Surfaces_Big("MeanPowerLoss2")
# Plot_Trace_Surfaces_Big("LPNormLoss2")
# Plot_Trace_Surfaces_Big("CrossEntropyLoss")
Plot_Curves("MeanPowerLoss2", show=True)
Plot_Curves("LPNormLoss2")
Plot_Curves("CrossEntropyLoss")
