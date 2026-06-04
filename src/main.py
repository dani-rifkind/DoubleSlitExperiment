import numpy as np
import matplotlib.pyplot as plt


def main():
    N = 500
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    slit_distance = 0.35
    wavelength = 0.08
    k = 2 * np.pi / wavelength

    source_y = -0.6

    r1 = np.sqrt((X + slit_distance / 2) ** 2 + (Y - source_y) ** 2)
    r2 = np.sqrt((X - slit_distance / 2) ** 2 + (Y - source_y) ** 2)

    wave = np.sin(k * r1) / np.sqrt(r1 + 1e-6) + np.sin(k * r2) / np.sqrt(r2 + 1e-6)
    intensity = wave**2

    plt.imshow(intensity, extent=[-1, 1, -1, 1], origin="lower")
    plt.colorbar(label="Intensity")
    plt.title("Double Slit Interference")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
