import numpy as np
import matplotlib.pyplot as plt
from scipy import fft


def f(u: np.array, x: np.array, j: int, Re: int) -> float:
    """RHS of the discretised equation in exercise 13

    Args:
        u (np.array): Solution u at some time.
        x (np.array): space discretisation.
        j (int): index of current point in space.
        Re (int): Reynolds number.

    Returns:
        float: RHS evaluated at current gridpoints.
    """
    h = x[1] - x[0]
    u0 = np.roll(u, 1)[j]
    u1 = u[j]
    u2 = np.roll(u, -1)[j]
    res = -(u1**2 - u0**2) / (2 * h) + 1 / Re * \
        (u2 - 2 * u1 + u0) / h**2 + np.sin(2 * np.pi * x[j])
    return res


def explicitEuler(f, Re: int, t: float = 1e-3, n: int = 100) -> np.array:
    """explicit Euler method

    Args:
        f (function): RHS of equation.
        Re (int): Reynolds nubmer.
        t (float, optional): time step size. Defaults to 1e-3.
        n (int, optional): grid points for discretisation. Defaults to 100.

    Returns:
        np.array: solution u on a grid.
    """
    m = int(1/t) + 1
    U = np.zeros((m, n))
    x = np.linspace(0, 1, n, endpoint=False)
    for i in range(m - 1):
        for j in range(n):
            U[i + 1, j] = U[i, j] + t * \
                f(U[i, :], x, j, Re)
    return U


def plotting(Re_list: list) -> list:
    """plot positive coefficients of the Fourier transformed
    solution for the discretisation at t = 0.6 using different
    values for Re.

    Args:
        Re_list (list): values of Re to solve pde for and plot the result.

    Returns:
        list: fourier transforms of solution for every Re
    """
    all_fft_U = list()
    for Re in Re_list:
        U = explicitEuler(f, Re)[600]
        fft_U = fft.fft(U)[0:len(U)//2]
        plt.loglog(np.abs(fft_U), label="Re = " + str(Re))
        all_fft_U.append(fft_U)
    plt.legend()
    plt.title("Fourier coefficients corresponding to positive frequencies")
    plt.savefig("toy_model.pdf", bbox_inches="tight", pad_inches=0.05)
    return all_fft_U


U = plotting([50, 200, 800])
