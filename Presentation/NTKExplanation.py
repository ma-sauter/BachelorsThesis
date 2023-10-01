# example.py

from manim import *

# or: from manimlib import *
from manim_slides import ThreeDSlide

from matplotlib import cm
import numpy as np

magma = cm.magma
magmacolors = magma(np.linspace(0, 1, 10))
magmacolorsformanim = []
for color in magmacolors:
    magmacolorsformanim.append(rgba_to_color(color))


#################################
# Surface function
# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)


def gauss(x, y, mx, my, sx, sy, Amplitude, C):
    return (
        -Amplitude * np.exp(-1 / sx * (x - mx) ** 2) * np.exp(-1 / sy * (y - my) ** 2)
        + C
    )


def Z_fn(X, Y):
    return (
        gauss(X, Y, 0, 0, 40, 40, 1, 2)
        + gauss(X, Y, 1, 1, 2, 2, -3, 1)
        + gauss(X, Y, -2, 1, 5, 2, -1.5, 1)
        + gauss(X, Y, -3, 0.5, 1, 1.3, -0.2, 1)
        + gauss(X, Y, -2, -1, 2, 2, 1, 1)
        + gauss(X, Y, 1, -1, 2, 2, -1, 0.5)
        + gauss(X, Y, -4, -4, 10, 10, 1.3, 1)
        + gauss(X, Y, -0.37, 1.7, 2, 2, -0.1, 0)
    )


def grad_fn(X, Y):
    e = 1e-4
    xgrad = (Z_fn(X + e, Y) - Z_fn(X - e, Y)) / 2 / e
    ygrad = (Z_fn(X, Y + e) - Z_fn(X, Y - e)) / 2 / e
    return np.array([xgrad, ygrad])


def gradient_descent(startx, starty, learning_rate, n_steps):
    xlist, ylist, zlist = (
        np.array([startx]),
        np.array([starty]),
        np.array([Z_fn(startx, starty) + 0.1]),
    )
    currentx, currenty = startx, starty

    for i in range(n_steps):
        currentgrad = grad_fn(currentx, currenty)
        currentx -= learning_rate * currentgrad[0]
        currenty -= learning_rate * currentgrad[1]
        xlist = np.append(xlist, currentx)
        ylist = np.append(ylist, currenty)
        zlist = np.append(zlist, Z_fn(currentx, currenty) + 0.1)
    return xlist, ylist, zlist


d1x, d1y, d1z = gradient_descent(0.3, 0.95, 1.5, 5)
d2x, d2y, d2z = gradient_descent(0.3, 0.95, 0.01, 1000)


F = Z_fn(X, Y)
#############################################


class NTKExplanation(ThreeDSlide):
    def construct(self):
        Slide_Heading = Tex(r"Neural Tangent Kernel", font_size=25).to_edge(UL)

        self.play(Write(Slide_Heading), run_time=0.3)
        self.next_slide()
        self.play(FadeOut(Slide_Heading), run_time=0.3)

        self.set_camera_orientation(theta=-100 * DEGREES, phi=70 * DEGREES)

        axes = ThreeDAxes(
            z_range=[np.min(F), np.max(F) * 1.5, (np.max(F) * 1.5 - np.min(F)) / 10]
        )

        surface = Surface(
            lambda u, v: axes.c2p(u, v, Z_fn(u, v)),
            u_range=[-5, 5],
            v_range=[-5, 5],
        )

        surface.set_fill_by_value(axes=axes, axis=2, colorscale=magmacolorsformanim)

        # def update_curve(d, dt):
        #    d.rotate_about_origin(dt, UP)

        # surface.add_updater(update_curve)

        self.play(FadeIn(axes, surface))

        self.begin_ambient_camera_rotation()

        self.wait(3)
        self.next_slide()

        sphere_list = VGroup(
            Sphere(
                center=axes.c2p(d1x[0], d1y[0], d1z[0]), radius=0.1, resolution=10
            ).set_color(GREEN)
        )
        for i in range(len(d1x) - 1):
            sphere_list.add(
                Line3D(
                    start=axes.c2p(d1x[i], d1y[i], d1z[i]),
                    end=axes.c2p(d1x[i + 1], d1y[i + 1], d1z[i + 1]),
                    color=GREEN,
                )
            )
            sphere_list.add(
                Sphere(
                    center=axes.c2p(d1x[i + 1], d1y[i + 1], d1z[i + 1]),
                    radius=0.1,
                    resolution=10,
                ).set_color(GREEN)
            )

        self.play(LaggedStart(Create(sphere_list), lag_ratio=0.3))
