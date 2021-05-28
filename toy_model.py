import numpy as np
import matplotlib.pyplot as plt
from scipy import fft


def f(u, x, j, Re):
    h = x[1] - x[0]
    u0 = np.roll(u, 1)[j]
    u1 = u[j]
    u2 = np.roll(u, -1)[j]
    res = -(u1**2 - u0**2) / (2 * h) + 1 / Re * \
        (u2 - 2 * u1 + u0) / h**2 + np.sin(2 * np.pi * x[j])
    return res


def explicitEuler(f, Re, t=1e-3, n=100):
    U = np.zeros((int(1/t), n))
    x = np.linspace(0, 1, n, endpoint=False)
    for i in range(len(U) - 1):
        for j in range(len(x)):
            U[i + 1, j] = U[i, j] + t * \
                f(U[i, :], x, j, Re)
    return U


def plotting(Re_list):
    for Re in Re_list:
        U = explicitEuler(f, Re)[599]
        fft_U = fft.fft(U)[0:len(U)//2]
        plt.loglog(np.abs(fft_U), label = "Re = " + str(Re))
    plt.legend()
    plt.title("Fourier coefficients corresponding to positive frequencies")
    plt.savefig("toy_model.pdf", bbox_inches="tight", pad_inches=0.05)


plotting([50, 200, 800])
