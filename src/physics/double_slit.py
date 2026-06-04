"""
Double-slit shared physics utilities.
Author contribution: Paul Völker implemented this project part.
"""

import numpy as np

H = 6.62607015e-34


def de_broglie_wavelength(mass, velocity):
    return H / (mass * velocity)


def double_slit_intensity(x, wavelength, slit_distance, screen_distance, slit_width):
    beta = np.pi * slit_width * x / (wavelength * screen_distance)
    alpha = np.pi * slit_distance * x / (wavelength * screen_distance)

    envelope = np.sinc(beta / np.pi) ** 2
    interference = np.cos(alpha) ** 2

    return envelope * interference


def normalized_probability(x, intensity):
    area = np.trapz(intensity, x)
    if area == 0:
        return intensity
    return intensity / area