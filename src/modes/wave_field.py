"""
Wave/probability-field mode.
Author contribution: Paul Völker implemented this project part.
"""

from functools import lru_cache

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from physics.double_slit import de_broglie_wavelength, double_slit_intensity

TITLE = "Wave Probability Field"

EXPLANATION = """
This mode shows massive-particle double-slit interference.

The screen plot uses the physical de Broglie wavelength:

λ = h / (m v)

The top view is pedagogical: the true wavelength and slit distance are far too
small to resolve on the full geometry scale, so the visible probability field
uses enlarged visual slit spacing. The final screen pattern remains physically
computed.
"""


def _screen_intensity(params, x_screen, wavelength):
    intensity = double_slit_intensity(
        x_screen,
        wavelength,
        params["slit_distance"],
        params["screen_distance"],
        params["slit_width"],
    )

    if params["which_path"]:
        beta = (
            np.pi
            * params["slit_width"]
            * x_screen
            / (wavelength * params["screen_distance"])
        )
        intensity = np.sinc(beta / np.pi) ** 2

    return intensity / (np.max(intensity) + 1e-12)


@lru_cache(maxsize=64)
def _cached_screen_pattern(
    mass,
    velocity,
    slit_distance,
    slit_width,
    screen_distance,
    screen_width,
    which_path,
):
    wavelength = de_broglie_wavelength(mass, velocity)

    x_screen = np.linspace(-screen_width, screen_width, 850)

    params = {
        "mass": mass,
        "velocity": velocity,
        "slit_distance": slit_distance,
        "slit_width": slit_width,
        "screen_distance": screen_distance,
        "screen_width": screen_width,
        "which_path": which_path,
    }

    final_intensity = _screen_intensity(params, x_screen, wavelength)

    return x_screen, final_intensity


@lru_cache(maxsize=64)
def _cached_geometry(
    source_distance,
    screen_distance,
    screen_width,
    slit_distance,
):
    z_min = -source_distance
    z_max = screen_distance

    # Better visual resolution than the ugly-fast version,
    # but still much cheaper than the original huge grid.
    z = np.linspace(z_min, z_max, 460)
    x = np.linspace(-screen_width, screen_width, 320)

    Z, X = np.meshgrid(z, x)

    visual_slit_sep = max(slit_distance, 0.12 * screen_width)
    visual_lambda = 0.036 * screen_width

    r_source = np.sqrt(X**2 + (Z + source_distance) ** 2)

    r1 = np.sqrt((X - visual_slit_sep / 2) ** 2 + Z**2)
    r2 = np.sqrt((X + visual_slit_sep / 2) ** 2 + Z**2)

    before_slit = Z <= 0
    after_slit_region = Z >= 0

    source_aperture = np.exp(-((X / (0.72 * screen_width)) ** 4))

    return (
        z,
        x,
        Z,
        X,
        r_source,
        r1,
        r2,
        before_slit,
        after_slit_region,
        source_aperture,
        visual_lambda,
        visual_slit_sep,
    )


def _draw_double_slit_barrier(ax, screen_width, slit_distance, slit_width):
    visible_slit_sep = max(slit_distance, 0.12 * screen_width)
    visible_slit_width = max(slit_width, 0.045 * screen_width)

    lower_slit = -visible_slit_sep / 2
    upper_slit = visible_slit_sep / 2

    gaps = [
        (lower_slit - visible_slit_width / 2, lower_slit + visible_slit_width / 2),
        (upper_slit - visible_slit_width / 2, upper_slit + visible_slit_width / 2),
    ]

    barrier_segments = [
        (-screen_width, gaps[0][0]),
        (gaps[0][1], gaps[1][0]),
        (gaps[1][1], screen_width),
    ]

    for y0, y1 in barrier_segments:
        if y1 > y0:
            ax.plot(
                [0, 0],
                [y0, y1],
                color="white",
                linewidth=7,
                solid_capstyle="butt",
                zorder=8,
            )

    for y0, y1 in gaps:
        ax.plot(
            [0, 0],
            [y0, y1],
            color="black",
            linewidth=11,
            solid_capstyle="butt",
            zorder=9,
        )


def render(params):
    source_distance = params["source_distance"]
    screen_distance = params["screen_distance"]
    screen_width = params["screen_width"]
    slit_distance = params["slit_distance"]
    slit_width = params["slit_width"]

    progress = float(params.get("progress", 0.0))
    progress = np.clip(progress, 0.0, 1.0)

    z_min = -source_distance
    z_max = screen_distance
    total_distance = source_distance + screen_distance
    front_z = z_min + progress * total_distance

    x_screen, final_intensity = _cached_screen_pattern(
        params["mass"],
        params["velocity"],
        slit_distance,
        slit_width,
        screen_distance,
        screen_width,
        params["which_path"],
    )

    (
        z,
        x,
        Z,
        X,
        r_source,
        r1,
        r2,
        before_slit,
        after_slit_region,
        source_aperture,
        visual_lambda,
        visual_slit_sep,
    ) = _cached_geometry(
        source_distance,
        screen_distance,
        screen_width,
        slit_distance,
    )

    # ---------- screen pattern ----------
    fig_screen, ax_screen = plt.subplots(figsize=(6.6, 3.0))

    if progress >= 0.995:
        ax_screen.plot(x_screen * 1e3, final_intensity, linewidth=2.2)
        ax_screen.fill_between(x_screen * 1e3, final_intensity, alpha=0.25)
    else:
        ax_screen.plot(x_screen * 1e3, np.zeros_like(x_screen), linewidth=2.2)

    ax_screen.set_title("Detection Pattern on Screen")
    ax_screen.set_xlabel("Screen position x [mm]")
    ax_screen.set_ylabel("Relative intensity")
    ax_screen.set_ylim(0, 1.05)
    ax_screen.grid(True, alpha=0.35)

    # ---------- top-view probability field ----------
    field = np.zeros_like(X)

    # Incoming source wave. Old waves stay visible.
    source_radius = max(front_z - z_min, 0.0)
    source_reached = r_source <= source_radius

    source_phase = 0.5 * (1.0 + np.cos(2 * np.pi * r_source / visual_lambda))
    field += 1.20 * source_phase * source_aperture * before_slit * source_reached

    # Outgoing post-slit field.
    if front_z > 0:
        after_slit = after_slit_region & (Z <= front_z)

        z_fraction = np.clip(Z / screen_distance, 0.0, 1.0)
        local_width = (0.065 + 0.935 * z_fraction) * screen_width

        x_mapped = X / (local_width + 1e-12) * screen_width

        screen_pattern = np.interp(
            x_mapped.ravel(),
            x_screen,
            final_intensity,
            left=0.0,
            right=0.0,
        ).reshape(X.shape)

        transverse_envelope = np.exp(-((X / (local_width + 1e-12)) ** 4))

        reached_1 = r1 <= front_z
        reached_2 = r2 <= front_z

        wave1 = np.cos(2 * np.pi * r1 / visual_lambda) * reached_1
        wave2 = np.cos(2 * np.pi * r2 / visual_lambda) * reached_2

        if params["which_path"]:
            outgoing = (wave1**2 + wave2**2) * transverse_envelope * after_slit
        else:
            interference_wave = wave1 + wave2
            outgoing = interference_wave**2 * transverse_envelope * after_slit
            outgoing *= 0.20 + 1.10 * screen_pattern**0.32

        field += 1.95 * outgoing

    field = field / (np.max(field) + 1e-12)
    field = field**0.35

    fig_top, ax_top = plt.subplots(figsize=(9.8, 5.2))

    ax_top.imshow(
        field,
        extent=[z_min, z_max, -screen_width, screen_width],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )

    ax_top.scatter(
        [-source_distance],
        [0],
        marker="*",
        s=220,
        color="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )

    ax_top.text(
        -source_distance,
        0.87 * screen_width,
        "source",
        color="white",
        ha="center",
        fontsize=10,
        weight="bold",
    )

    _draw_double_slit_barrier(ax_top, screen_width, slit_distance, slit_width)

    ax_top.text(
        0,
        -0.88 * screen_width,
        "slits enlarged for visibility",
        color="white",
        ha="center",
        fontsize=9,
        weight="bold",
    )

    ax_top.axvline(front_z, color="cyan", linestyle="--", linewidth=2.2, zorder=7)

    ax_top.axvline(screen_distance, color="red", linewidth=3.0, zorder=8)
    ax_top.text(
        screen_distance,
        0.87 * screen_width,
        "screen",
        color="red",
        ha="center",
        fontsize=10,
        weight="bold",
    )

    if progress >= 0.995:
        pattern_scale = final_intensity**0.35 * screen_distance * 0.22
        ax_top.plot(
            screen_distance + pattern_scale,
            x_screen,
            color="red",
            linewidth=3.0,
            zorder=9,
        )

    ax_top.set_title("Top View: Probability Field and Interference", fontsize=13)
    ax_top.set_xlabel("Propagation direction z [m]")
    ax_top.set_ylabel("Transverse position x [m]")
    ax_top.set_xlim(z_min, z_max + 0.24 * screen_distance)
    ax_top.set_ylim(-screen_width, screen_width)

    return fig_top, fig_screen