import numpy as np
import matplotlib.pyplot as plt
import pyarma as pa

plt.rc("font", size=17)
plt.rc('legend', frameon=False)

### Problem 6: Plotting scaling with N
N = pa.mat(); N.load("N.dat")
iterations = pa.mat(); iterations.load("iterations.dat")
plt.plot(N, iterations, label="Data")
plt.plot(N, 1.7*np.array(N)**2, "--", label=r"1.7x$^2$", alpha=0.6)
plt.plot(N, 1.8*np.array(N)**2, "--", label=r"1.8x$^2$", alpha=0.6)
plt.plot(N, 1.9*np.array(N)**2, "--", label=r"1.9x$^2$", alpha=0.6)
plt.title("Numer of iterations required as function of points N")
plt.xlabel("Number of points N")
plt.ylabel("Iterations required")
plt.legend()
plt.show()

### Problem 7: Plotting solutions
eigenvalues = pa.mat(); eigenvalues.load("eigenvalues.dat")
eigenvectors = pa.mat(); eigenvectors.load("eigenvectors.dat")
analytical_eigenvectors = pa.mat(); analytical_eigenvectors.load("analytical_eigenvectors.dat")

eigenvectors = np.array(eigenvectors).T #making vectors into numpy arrays for simplicity
analytical_eigenvectors = np.array(analytical_eigenvectors).T

N = len(eigenvectors[0]) #number of points in system (middle)
x = np.linspace(0, 1, N+2) #N+2 to include endpoints, n=N+1, x=x0, x1, ..., xn == n+1 points
for i in range(len(eigenvectors)):
    sol_num = np.array([0] + list(eigenvectors[i]) + [0]) #adding endpoints, numerical solution
    sol_anl = np.array([0] + list(analytical_eigenvectors[i]) + [0]) #analytical solution
    plt.plot(x, sol_num, label=r"$\lambda_i = ${:.2f}".format(eigenvalues[i]))
    plt.plot(x, sol_anl, "--k", alpha=0.5)
plt.title("Three numerical solutions with lowest eigenvalues, n = {} steps".format(N-1))
plt.plot([], [], "--k", label="Analytical solutions") #hack to get single label for all
plt.xlabel(r"$\hat{x}$")
plt.ylabel(r"$\vec{v}$")
plt.legend()
plt.show()