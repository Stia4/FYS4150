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
x, u = read_x_y("xu.txt") #exact solution
plt.plot(x, u, label="u(x), n=10001")

x1, v1 = read_x_y("xv_n10.txt") #numerical solutions
x2, v2 = read_x_y("xv_n100.txt")
x3, v3 = read_x_y("xv_n1000.txt")
x4, v4 = read_x_y("xv_n10000.txt")
plt.plot(x1, v1, label="v(x), n=11")
plt.plot(x2, v2, label="v(x), n=101")
plt.plot(x3, v3, label="v(x), n=1001")
plt.plot(x4, v4, label="v(x), n=10001")
plt.title("Comparison of exact and numerical solutions")
plt.legend()
plt.savefig("comparison.pdf", format="pdf")
plt.clf() #clear figure

# ========= Problem 8 ==========
xi = pa.mat(); xi.load("xi.data") #loading saved data
ui = pa.mat(); ui.load("ui.data")
vi = pa.mat(); vi.load("vi.data")
xi = np.array(xi) #numpy arrays are easier to use here
ui = np.array(ui)
vi = np.array(vi)

# Absolute error
delta_i = abs(ui - vi)
for i in range(len(delta_i[0, :])):
    N = sum(~np.isnan(delta_i[:, i])) #number of non-nan elements
    x = xi[:N, i] #each column is a plot, grabbing only non-nan
    di = delta_i[:N, i]    
    plt.plot(x, di, label=r"N = {:d}, dx = {:.0e}".format(N, 1/(N-1)))

plt.title(r"Absolute error $\Delta$ for different number of points")
plt.xlabel("x")
plt.ylabel(r"$\Delta$")
plt.legend()
plt.yscale('log')
plt.savefig("abs_err.pdf", format="pdf")
plt.clf()

# Relative error
ui_nan = ui; ui_nan[ui == 0] = np.nan #replacing 0 with nan, due to divide by 0 in eps
eps_i = abs((ui - vi)/ui_nan)
for i in range(len(eps_i[0, :])):
    N = sum(~np.isnan(eps_i[:, i])) #number of non-nan elements
    x = xi[:N, i]
    ei = eps_i[:N, i]
    print("Maximum relative error: {:.3f} for N = {:d}".format(np.nanmax(ei), N))
    plt.plot(x, ei, label=r"N = {:d}, dx = {:.0e}".format(N+1, 1/N)) #N+1 due to ui_nan

plt.title(r"Relative error $\epsilon$ for different number of points")
plt.xlabel("x")
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.yscale('log')
plt.savefig("rel_err.pdf", format="pdf")
plt.clf()