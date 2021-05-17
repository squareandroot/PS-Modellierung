import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import newton


def fun(x, f, z):
    return np.vstack((f[1], f[2], -0.5 * f[2] * f[0]))


def bc(fa, fb, z):
    return np.array([fa[0], fa[1], z - fa[2], 1.0 - fb[1]], dtype=float)


x = np.linspace(0, 10, 5)
y = np.zeros((3, x.size))
sol = solve_bvp(fun, bc, x, y, p=[0.5])
x_plot = np.linspace(0, 10, 100)
y_plot = sol.sol(x_plot)[1]
plt.plot(x_plot, y_plot)
plt.xlabel(r"$\eta$")
plt.ylabel(r"$f'(\eta)$")
plt.title(r"Solution for the ideal parameter $z = $" + str(round(sol.p[0], 4)))
plt.grid()
plt.savefig("plot_blasius_equation.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()


def f(x):
    return sol.sol(x)[1] - 0.99


for_boundary_layer = newton(f, 6)


# Implementation with explicit Euler method and bisection to determine z

def explicitEuler(z, h, N):
    t = 0 + np.arange(N + 1) * h
    x0 = np.zeros(N + 1)
    x0[0] = 0
    x1 = np.zeros(N + 1)
    x1[0] = 0
    x2 = np.zeros(N + 1)
    x2[0] = z
    for n in range(N):
        x0[n+1] = x0[n] + h * x1[n]
        x1[n+1] = x1[n] + h * x2[n]
        x2[n+1] = x2[n] - h * x0[n] * x2[n] / 2
    return x1[N], t, x1


def bisection(z0, z1, h, N, tol=1e-8):
    while abs(z0 - z1) > tol:
        c = 0.5 * (z0 + z1)
        y_c = explicitEuler(c, h, N)[0]
        if(y_c > 1 + tol):
            z1 = c
        elif(y_c < 1 + tol):
            z0 = c
        else:
            return c
    return 0.5 * (z0 + z1)


z = bisection(0, 10, h=0.01, N=1000)

sol, x_plot, y_plot = explicitEuler(z, h=0.01, N=1000)

plt.plot(x_plot, y_plot)
plt.xlabel(r"$\eta$")
plt.ylabel(r"$f'(\eta)$")
plt.title(r"Solution for the ideal parameter $z = $" + str(round(z, 4)))
plt.grid()
plt.show()
