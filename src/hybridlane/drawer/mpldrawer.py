# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import warnings
from typing import Literal

import matplotlib.colors as mc
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial
from pennylane.drawer.mpldrawer import MPLDrawer, _to_tuple


class HybridMPLDrawer(MPLDrawer):
    _icon_width = 0.5
    _icon_face_alpha = 0.3
    _qumode_padding = 0.05

    _instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store instances so we can retrieve them in _draw_icons
        self._instances.append(self)

    def wire_icons(
        self, icons: list[Literal["qumode", "qubit"]], colors: list | None = None
    ):
        # Shift the left side over to make room for the icons
        left = self._ax.get_xlim()[0]
        left -= self._icon_width + 0.25  # extra padding
        self._ax.set_xlim(left=left)

        icon_x = left + self._icon_width / 2 + 0.25

        colors = colors or [plt.rcParams["lines.color"]] * len(icons)
        for wire, (icon, color) in enumerate(zip(icons, colors)):
            if icon not in ("qumode", "qubit"):
                raise ValueError(f"Unknown icon type {icon}")

            options = {"color": color}

            match icon:
                case "qumode":
                    self._qumode_icon(icon_x, wire, options=options)
                case "qubit":
                    self._qubit_icon(icon_x, wire, options=options)

    def _qumode_icon(self, icon_x, wire, options: dict | None = None):
        # For a parabola with the icon centered at x0, y0, and width w, h, we have
        #  - the base of the parabola f(x0) = y0 - h/2
        #  - the top edges of the parabola f(x0 +/- w/2) = y0 + h/2
        options = options or {}
        x = icon_x
        y = wire
        w, h = self._icon_width, self._icon_width
        color = options.get("color", plt.rcParams["lines.color"])

        # Draw containing box
        bg_color = icon_face_color(color, self._icon_face_alpha)
        rect = patches.Rectangle(
            (x - w / 2, y - h / 2),
            width=w,
            height=h,
            edgecolor="black",
            facecolor=bg_color,
            linewidth=1,
        )
        self._ax.add_patch(rect)

        # Draw the quantum harmonic oscillator
        w -= 2 * self._qumode_padding
        h -= 2 * self._qumode_padding

        poly = Polynomial.fit(
            [x, x + w / 2, x - w / 2], [y + h / 2, y - h / 2, y - h / 2], 2
        )

        # Plot the parabola using black line color
        x_vals = np.linspace(x - w / 2, x + w / 2, num=15)
        y_vals = poly(x_vals)
        plt.plot(x_vals, y_vals, color="black", zorder=4)

        # Now draw each energy level using `color`
        levels = 3
        delta = h / (levels + 1)
        for n in range(1, levels + 1):
            y_pos = y + h / 2 - n * delta
            dx = _parabola_width(poly, y_pos)
            plt.plot([x - dx / 2, x + dx / 2], [y_pos, y_pos], color=color, zorder=3)

    def _qubit_icon(self, icon_x, wire, options: dict | None = None):
        # Draw a circle with an = inside it
        options = options or {}
        x = icon_x
        y = wire
        r = self._icon_width / 2
        color = options.get("color", plt.rcParams["lines.color"])

        # Draw containing circle
        bg_color = icon_face_color(color, self._icon_face_alpha)
        rect = patches.Circle(
            (x, y),
            radius=r,
            edgecolor="black",
            facecolor=bg_color,
            linewidth=1,
        )
        self._ax.add_patch(rect)

        # Now add the two lines inside at y = -r/3, r/3
        w = 0.6 * _circle_width(r, r / 3)
        plt.plot([x - w / 2, x + w / 2], [y - r / 3, y - r / 3], color=color)
        plt.plot([x - w / 2, x + w / 2], [y + r / 3, y + r / 3], color=color)

    def z_conditional(
        self, layer, wires, wires_target=None, control_values=None, options=None
    ):
        if options is None:
            options = {}

        wires_ctrl = _to_tuple(wires)
        wires_target = _to_tuple(wires_target)
        if control_values is not None:
            control_values = _to_tuple(control_values)

        wires_all = wires_ctrl + wires_target
        min_wire = min(wires_all)
        max_wire = max(wires_all)

        if len(wires_target) > 1:
            min_target, max_target = min(wires_target), max(wires_target)
            if any(min_target < w < max_target for w in wires_ctrl):
                warnings.warn(
                    "Some conditional indicators are hidden behind an operator. Consider re-ordering "
                    "your circuit wires to ensure all control indicators are visible.",
                    UserWarning,
                )

        line = plt.Line2D((layer, layer), (min_wire, max_wire), **options)
        self._ax.add_line(line)

        if control_values is None:
            for wire in wires_ctrl:
                self._cond_diamond(layer, wire, options=options)
        else:
            if len(control_values) != len(wires_ctrl):
                raise ValueError("`control_values` must be the same length as `wires`")
            for wire, control_on in zip(wires_ctrl, control_values):
                if control_on:
                    self._cond_diamond(layer, wire, options=options)
                else:
                    self._condo_diamond(layer, wire, options=options)

    def _cond_diamond(self, layer, wires, options=None):
        if options is None:
            options = {}
        if "color" not in options:
            options["color"] = plt.rcParams["lines.color"]
        if "zorder" not in options:
            options["zorder"] = 3

        poly = patches.RegularPolygon(
            (layer, wires), 4, radius=self._ctrl_rad, orientation=0, **options
        )
        self._ax.add_patch(poly)

    def _condo_diamond(self, layer, wires, options=None):
        new_options = _open_diamond_options_process(options)

        poly = patches.RegularPolygon(
            (layer, wires),
            4,
            radius=self._ctrl_rad,
            orientation=0,
            **new_options,
        )
        self._ax.add_patch(poly)


def _open_diamond_options_process(options):
    options = options or {}

    new_options = options.copy()
    if "color" in new_options:
        new_options["facecolor"] = plt.rcParams["axes.facecolor"]
        new_options["edgecolor"] = options["color"]
        new_options["color"] = None
    else:
        new_options["edgecolor"] = plt.rcParams["lines.color"]
        new_options["facecolor"] = plt.rcParams["axes.facecolor"]

    if "linewidth" not in new_options:
        new_options["linewidth"] = plt.rcParams["lines.linewidth"]
    if "zorder" not in new_options:
        new_options["zorder"] = 3

    return new_options


def _parabola_width(poly: Polynomial, y: float):
    new_poly = poly - [y]
    x0, x1 = new_poly.roots().real
    return np.abs(x0 - x1)


def _circle_width(r, y):
    return 2 * np.sqrt(r**2 - y**2)


def icon_face_color(color, alpha):
    color = _adjust_alpha(color, alpha)
    color = _blend(color, "white")
    return tuple(color)


def _adjust_alpha(color, alpha):
    c = mc.cnames.get(color, color)
    c = mc.to_rgb(c)
    return (*c, np.clip(alpha, 0, 1))


def _blend(color_rgba, bg_color):
    bg_c = mc.cnames.get(bg_color, bg_color)
    bg = np.array(mc.to_rgb(bg_c))
    c = np.array(color_rgba[:3])
    alpha = color_rgba[-1]

    blended = alpha * c + (1 - alpha) * bg
    return np.clip(blended, 0, 1).tolist()
