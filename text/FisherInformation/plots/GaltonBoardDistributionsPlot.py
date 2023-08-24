from re import T
import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import quad
from math import factorial
from matplotlib.lines import Line2D
from matplotlib import cm


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathrsfs}",
    "font.family": "Helvetica",
    "font.size": "15"
})


def binomial(n,p):
    liste = []
    for k in range(n+1):
        liste.append(factorial(n)/factorial(k)/factorial(n-k) * p**(k)*(1-p)**(n-k))
    return np.array(liste)
    
def galton(theta,n):
    assert theta in np.arange(n+1)
    assert n%2 == 0
    b_list = binomial(n,0.5)
    b_split_list = np.split(b_list, [int(np.max([-theta+n/2,0])),int(np.min([n+1, (n+1) - (theta-n/2)]))])
    final_list = np.append(np.zeros(int(np.max([theta-n/2,0]))), b_split_list[1])
    final_list = np.append(final_list, np.zeros(int(np.max([-theta+n/2,0]))))
    for i, wert in enumerate(b_split_list[0][::-1]):
        final_list[i] += wert
    for i, wert in enumerate(b_split_list[2]):
        final_list[-i-1] += wert
    return final_list

n_plots = 4
scale = 3
fig, axes = plt.subplots(2,n_plots, figsize = (3*scale, scale))

for i in range(n_plots):
    theta = 2+3*i
    data = galton(theta,14)
    ax = axes[1,i]
    ax.set_xticks([])
    ax.set_yticks([])
    normalized_data = data / np.max(data) / 2
    ax.bar(np.arange(len(data)), data, color = cm.magma(normalized_data))

    #Theta axis shenanigans
    tax = axes[0,i]
    tax.set_xlim(ax.get_xaxis().get_view_interval())
    tax.tick_params(axis="x",direction="in", pad=-26)
    tax.set_xticks([theta], labels = [f"$\\theta = {theta}$"])
    tax.tick_params(direction = 'inout', length = 10, width = 2)
    tax.xaxis.set_label_position('top')



    
    tax.set_frame_on(False)
    tax.axes.get_yaxis().set_visible(False)
    xmin, xmax = tax.get_xaxis().get_view_interval()
    ymin, ymax = tax.get_yaxis().get_view_interval()
    tax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))

plt.subplots_adjust(wspace=0, hspace = 0.07)
#plt.show()
plt.savefig("GaltonDistributionsPlot.pdf", bbox_inches='tight')

        
    