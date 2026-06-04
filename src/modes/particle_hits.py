"""
Particle/discrete mode.
Author contribution: Ofek implemented this project part.
"""

from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.double_slit import de_broglie_wavelength, double_slit_intensity

TITLE = "Discrete Particles"

EXPLANATION = """
This mode demonstrates the particle nature of the double-slit experiment. 

Instead of a continuous wave, massive particles travel through the slits and strike the screen as discrete entities. Notice how individual impacts seem random at first, but over time, they build up to match the exact theoretical probability distribution determined by their de Broglie wavelength.
"""

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
            ax.plot([0, 0], [y0, y1], color="white", linewidth=7, solid_capstyle="butt", zorder=8)

    for y0, y1 in gaps:
        ax.plot([0, 0], [y0, y1], color="black", linewidth=11, solid_capstyle="butt", zorder=9)


@lru_cache(maxsize=64)
def _generate_particles(mass, velocity, slit_dist, slit_width, screen_dist, screen_width, which_path, num_particles=3000):
    """Pre-calculates all particle strikes and flight timings."""
    wavelength = de_broglie_wavelength(mass, velocity)
    x_vals = np.linspace(-screen_width, screen_width, 2500)

    intensity = double_slit_intensity(x_vals, wavelength, slit_dist, screen_dist, slit_width)
    
    if which_path:
        beta = np.pi * slit_width * x_vals / (wavelength * screen_dist)
        intensity = np.sinc(beta / np.pi) ** 2

    # Normalize to create a PDF and then a CDF for inverse transform sampling
    pdf = intensity / (np.sum(intensity) + 1e-12)
    cdf = np.cumsum(pdf)

    # Sample random hits based on the distribution
    random_probs = np.random.rand(num_particles)
    hit_indices = np.searchsorted(cdf, random_probs)
    hit_indices = np.clip(hit_indices, 0, len(x_vals) - 1)
    hits = x_vals[hit_indices]

    # Generate staggered start times for the animation
    start_times = np.random.uniform(0.0, 0.85, num_particles)
    
    # Assign each particle randomly to the top or bottom slit for visual pathing
    slit_choice = np.random.choice([-1, 1], num_particles)

    # Sort chronologically so Streamlit's progress slider reveals them smoothly
    sort_idx = np.argsort(start_times)
    return hits[sort_idx], start_times[sort_idx], slit_choice[sort_idx]


def render(params):
    source_distance = params["source_distance"]
    screen_distance = params["screen_distance"]
    screen_width = params["screen_width"]
    slit_distance = params["slit_distance"]
    slit_width = params["slit_width"]

    progress = float(params.get("progress", 0.0))
    progress = np.clip(progress, 0.0, 1.0)

    # Fetch cached particle data to prevent jitter between frames
    hits, start_times, slit_choice = _generate_particles(
        params["mass"],
        params["velocity"],
        slit_distance,
        slit_width,
        screen_distance,
        screen_width,
        params["which_path"]
    )

    flight_duration = 0.15 

    # Determine state of each particle
    has_spawned = progress >= start_times
    has_hit = progress >= (start_times + flight_duration)
    in_flight = has_spawned & ~has_hit

    # ---------- Top View Animation ----------
    fig_top, ax_top = plt.subplots(figsize=(9.8, 5.2))
    ax_top.set_facecolor("#111111") 

    _draw_double_slit_barrier(ax_top, screen_width, slit_distance, slit_width)

    # Draw Source
    ax_top.scatter([-source_distance], [0], marker="*", s=220, color="white", edgecolors="black", zorder=10)
    ax_top.text(-source_distance, 0.87 * screen_width, "source", color="white", ha="center", fontsize=10, weight="bold")

    # Draw Screen
    ax_top.axvline(screen_distance, color="red", linewidth=3.0, zorder=8)
    ax_top.text(screen_distance, 0.87 * screen_width, "screen", color="red", ha="center", fontsize=10, weight="bold")

    visual_slit_sep = max(slit_distance, 0.12 * screen_width)

    # Plot particles currently flying
    if np.any(in_flight):
        flight_progress = (progress - start_times[in_flight]) / flight_duration
        z_total = source_distance + screen_distance
        z_current = -source_distance + flight_progress * z_total

        x_current = np.zeros_like(z_current)

        # Path logic before passing the slits
        mask_before = z_current < 0
        if np.any(mask_before):
            frac = (z_current[mask_before] + source_distance) / source_distance
            target_x = slit_choice[in_flight][mask_before] * (visual_slit_sep / 2)
            x_current[mask_before] = frac * target_x

        # Path logic after passing the slits
        mask_after = z_current >= 0
        if np.any(mask_after):
            frac = z_current[mask_after] / screen_distance
            start_x = slit_choice[in_flight][mask_after] * (visual_slit_sep / 2)
            target_x = hits[in_flight][mask_after]
            x_current[mask_after] = start_x + frac * (target_x - start_x)

        ax_top.scatter(z_current, x_current, color="cyan", s=20, zorder=11, alpha=0.9)

    # Plot particles that have already arrived at the screen
    if np.any(has_hit):
        arrived_hits = hits[has_hit]
        ax_top.scatter(np.full_like(arrived_hits, screen_distance), arrived_hits, color="orange", s=6, alpha=0.35, zorder=12)

    ax_top.set_title("Top View: Discrete Particle Trajectories", fontsize=13)
    ax_top.set_xlabel("Propagation direction z [m]")
    ax_top.set_ylabel("Transverse position x [m]")
    ax_top.set_xlim(-source_distance, screen_distance + 0.24 * screen_distance)
    ax_top.set_ylim(-screen_width, screen_width)

    # ---------- Screen Pattern ----------
    fig_screen, ax_screen = plt.subplots(figsize=(6.6, 3.0))

    # Calculate underlying theoretical envelope
    x_vals = np.linspace(-screen_width, screen_width, 1000)
    wavelength = de_broglie_wavelength(params["mass"], params["velocity"])
    intensity = double_slit_intensity(x_vals, wavelength, slit_distance, screen_distance, slit_width)
    
    if params["which_path"]:
        beta = np.pi * slit_width * x_vals / (wavelength * screen_distance)
        intensity = np.sinc(beta / np.pi) ** 2
        
    intensity_norm = intensity / (np.max(intensity) + 1e-12)

    ax_screen.plot(x_vals * 1e3, intensity_norm, color="red", linewidth=2.0, alpha=0.5, label="Theoretical Probability")

    # Overlay live histogram of impacts
    if np.any(has_hit):
        arrived_hits = hits[has_hit] * 1e3 
        hist_counts, bin_edges = np.histogram(arrived_hits, bins=80, range=(-screen_width*1e3, screen_width*1e3))
        
        if np.max(hist_counts) > 0:
            normalized_hist = hist_counts / np.max(hist_counts)
            ax_screen.bar(bin_edges[:-1], normalized_hist, width=np.diff(bin_edges), align="edge", color="orange", alpha=0.75, label="Accumulated Impacts")

    ax_screen.set_title("Detection Pattern on Screen")
    ax_screen.set_xlabel("Screen position x [mm]")
    ax_screen.set_ylabel("Relative Intensity")
    ax_screen.set_ylim(0, 1.05)
    ax_screen.grid(True, alpha=0.35)
    ax_screen.legend(loc="upper right")

    return fig_top, fig_screen