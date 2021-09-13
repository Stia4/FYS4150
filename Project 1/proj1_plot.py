import numpy as np
import matplotlib.pyplot as plt
import pyarma as pa

def read_x_y(filename):
    datafile = open(filename)
    datafile.readline() #skipping header line

    #read string, split into words, save as numbers in array
    numbers = np.array(datafile.read().split(), dtype=float)
    x = numbers[0::2] #every other entry in array, starting at 0
    y = numbers[1::2] #starting at 1

    datafile.close()
    return x, y

# ====== Problems 2 and 7 ======
fig, ax = plt.subplots(1, 2)
x, u = read_x_y("xu.txt")
ax[0].plot(x, u)
x, v = read_x_y("xv.txt")
ax[1].plot(x, v)
plt.show() #quick comparation between analytical and numerical for set n

# ========= Problem 8 ==========
xi = pa.mat(); xi.load("xi.txt") #loading saved data
ui = pa.mat(); ui.load("ui.txt")
vi = pa.mat(); vi.load("vi.txt")
xi = np.array(xi) #numpy arrays are easier to use here
ui = np.array(ui)
vi = np.array(vi)

# Comparison plots
for i in range(len(xi[0, :])):
    N = sum(~np.isnan(xi[:, i])) #number of non-nan elements
    x = xi[:N, i] #each column is a plot, grabbing only non-nan
    u = ui[:N, i]
    v = vi[:N, i]

    plt.plot(x, u, label=r"Analytical u($x_i$)")
    plt.plot(x, v, label=r"Numerical v($x_i$)/n$^2$")
    plt.title("Analytical VS Numerical for N = {:d} points, dx = {:.0e}".format(N, 1/(N-1)))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Absolute error
delta_i = abs(ui - vi)
for i in range(len(delta_i[0, :])):
    N = sum(~np.isnan(delta_i[:, i])) #number of non-nan elements
    x = xi[:N, i]
    di = delta_i[:N, i]    
    plt.plot(x, di, label=r"N = {:d}, dx = {:.0e}".format(N, 1/(N-1)))

plt.title("Absolute error for different number of points")
plt.xlabel("x")
plt.ylabel(r"$\Delta$")
plt.legend()
plt.yscale('log')
plt.show()

# Relative error
ui_nan = ui; ui_nan[ui == 0] = np.nan #replacing 0 with nan, due to divide by 0 in eps
eps_i = abs((ui - vi)/ui_nan)
for i in range(len(eps_i[0, :])):
    N = sum(~np.isnan(eps_i[:, i])) #number of non-nan elements
    x = xi[:N, i]
    ei = eps_i[:N, i]
    plt.plot(x, ei, label=r"N = {:d}, dx = {:.0e}".format(N+1, 1/N)) #N+1 due to ui_nan

plt.title("Relative error for different number of points")
plt.xlabel("x")
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.yscale('log')
plt.show()

# Max of relative error
