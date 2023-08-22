import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathrsfs}",
    "font.family": "Helvetica",
    "font.size": "15"
})

learning_rate = 0.3
plotfactor = 1

def f(x):
    a, b, c, d, f = 1.4,2.6,-1.2,-2.7, 1
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + f
def df(x):
    a, b, c, d, f = 1.4,2.6,-1.2,-2.7, 1
    return 5*a*x**4 + 4*b*x**3 + 3*c*x**2 + 2*d*x**1
def tangent(x,a):
    return x*df(a) + f(a)-a*df(a)
def draw_dot_and_arrow(dotx, arrowcolor='black'):
    x_list1 = np.linspace(dotx-0.5,dotx+0.5, 100)
    plt.plot(dotx, f(dotx), "+", color = arrowcolor, markersize = plotfactor*10, markeredgewidth = plotfactor*3)
    plt.plot(x_list1, tangent(x_list1,dotx), "--", color = "gray", linewidth = plotfactor*0.7)
    angle = np.arctan(df(dotx))
    ax.annotate("", xytext=(dotx, f(dotx)),
                xy=(dotx + learning_rate*(-df(dotx)), tangent(dotx + learning_rate*(-df(dotx)), dotx)),
                arrowprops=dict(lw = plotfactor*1.5,shrinkB=0,color = arrowcolor, arrowstyle="->, head_width=0.3"))
    
    plt.plot([dotx + learning_rate*(-df(dotx)), dotx + learning_rate*(-df(dotx))],
             [tangent(dotx + learning_rate*(-df(dotx)), dotx), f(dotx + learning_rate*(-df(dotx)))],
             "--", color = arrowcolor)
    plt.plot(dotx + learning_rate*(-df(dotx)), f(dotx + learning_rate*(-df(dotx))),
             "o", markersize = plotfactor*8, color = arrowcolor)



x_list = np.linspace(-2,1.5,1000)

fig, ax = plt.subplots()
plt.plot(x_list, f(x_list), color = 'black')
draw_dot_and_arrow(0.25, arrowcolor='crimson')
draw_dot_and_arrow(-0.85, arrowcolor='seagreen')

height = 3
fig.set_figheight(height)
fig.set_figwidth(2.5*height)
plt.ylim(-0.3,1.8)
plt.annotate(r"$\mathscr{L}$", xy= (0.8,1.5))

#X axis shenanigans
plt.xlabel(r"$\theta_1$")
plt.xticks([])
ax.set_frame_on(False)
ax.axes.get_yaxis().set_visible(False)
xmin, xmax = ax.get_xaxis().get_view_interval()
ymin, ymax = ax.get_yaxis().get_view_interval()
ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))


#plt.show()
plt.savefig("sgd_plot.pdf", bbox_inches='tight')