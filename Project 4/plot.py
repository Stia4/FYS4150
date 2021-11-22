import numpy as np
import matplotlib.pyplot as plt
import pyarma as pa
from scipy.signal import lfilter
from scipy.stats import linregress
plt.rc('legend', frameon=False)
plt.rc('figure', figsize=(10, 6))
plt.rc('font', size=15)

def Sol_2x2():
    ### Not used in report, only for early debugging
    Rel_err = pa.mat(); Rel_err.load("Rel_err_2x2.dat"); Rel_err = np.array(Rel_err)
    T = np.linspace(2.1, 2.4, 1000)
    labels = [r"<$\epsilon$>", r"<$\epsilon^2$>", r"<|m|>", r"<m$^2$>", r"C$_{\rm V}$", r"$\chi$"]
    plt.figure(figsize=(25,9))
    for i in range(6):
        plt.plot(T, Rel_err[:, i], label=labels[i])
    plt.title("Relative error")
    plt.xlabel(r"T [J/k$_{\rm B}$]")
    plt.legend()
    plt.show()

    A = pa.mat(); A.load("Results_2x2.dat"); A = np.array(A)
    T = np.linspace(2.1, 2.4, 1000)
    titles = [r"<$\epsilon$>", r"<$\epsilon^2$>", r"<|m|>", r"<m$^2$>", r"C$_{\rm V}$", r"$\chi$"]

    fig, ax = plt.subplots(2, 3, figsize=(25, 9))
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(T, A[:, 3*i+j])
            ax[i, j].set_title(titles[3*i+j])
            ax[i, j].set_xlabel(r"T [J/k$_{\rm B}$]")
    plt.tight_layout()
    plt.show()

def load_mat(path):
    """
    Loads armadillo matrices at specified file path
    """
    M = pa.mat()    # initialize matrix
    M.load(path)    # load from file
    M = np.array(M) # convert to numpy matrix
    return M

def Histogram():
    T1, T2 = load_mat("data/Histogram.dat")
    fig, ax = plt.subplots(2, 1, figsize=(9, 12))
    x = np.arange(-800, 800, 4) / 400
    ax[0].bar(x, T1/sum(T1), width=4/400, align='edge', color="C0")
    ax[0].set_xlim(-2.1, -1.9)
    ax[1].bar(x, T2/sum(T2), width=4/400, align='edge', color="C1")
    ax[1].set_xlim(-2, -0.5)
    ax[0].set_xlabel(r"$\epsilon/J$")
    ax[1].set_xlabel(r"$\epsilon/J$")
    ax[0].set_title(r"$T = 1 J/k_B$")
    ax[1].set_title(r"$T = 2.4 J/k_B$")
    #plt.suptitle(r"Normalized histograms of $10^5$ samples of $\epsilon$")
    plt.savefig("fig/Histograms.pdf", format="pdf")
    plt.clf()

def plot_results():
    T = np.linspace(2.1, 2.4, 100)
    titles = [r"<$\epsilon$>", r"<|m|>", r"C$_{\rm V}$", r"$\chi$"]
    fig, ax = plt.subplots(2, 2, figsize=(9, 12))
    data = ["20", "40", "60", "80", "100"]
    for dat in data:
        A = load_mat("data/Results_"+dat+"x"+dat+"_500k.dat")[:, [0, 2, 4, 5]]
        for i in range(2):
            for j in range(2):
                ax[j, i].plot(T, A[:, 2*i+j], label="L = "+dat)
    for i in range(2):
        for j in range(2):
            ax[j, i].set_title(titles[2*i+j])
            ax[j, i].set_xlabel(r"T [J/k$_{\rm B}$]")
            ax[j, i].legend()
    plt.tight_layout()
    plt.savefig("fig/Results.pdf", format="pdf")
    plt.clf()

def Cv_peaks():
    T = np.linspace(2.1, 2.4, 100)
    data = ["20", "40", "60", "80", "100"]
    Cv = np.zeros((len(data), len(T)))
    for i, N in enumerate(data):
        Cv[i] = load_mat("data/Results_"+N+"x"+N+"_500k.dat")[:, 4]
    
    maxpoint = np.zeros((len(data), 2))
    for i, y in enumerate(Cv):
        n = 3
        w = lfilter([1.0 / n]*n, 1, y)
        plt.plot(T, w, color="C"+str(i), label="L = "+data[i])
        plt.plot(T, y, "x", color="C"+str(i))
        maxpoint[i] = [T[np.argmax(y)], max(y)]
        plt.plot(T[np.argmax(y)], max(y), "oC"+str(i))
    
    slope, intercept = linregress(maxpoint[:, 0], maxpoint[:, 1])[:2]
    #plt.plot(maxpoint[:, 0], intercept + slope*maxpoint[:, 0], "k--")

    L = np.array(data, dtype=float)
    slope_L, intercept_L = linregress(L, maxpoint[:, 0])[:2]
    print("Tc =", intercept_L, ", a =", slope_L)

    plt.plot([], [], "xk", label="Data")
    plt.plot([], [], "k", label="Low pass filter")
    plt.plot([], [], "ok", label="Max datapoint")
    plt.legend()
    plt.ylim(0.9, 2.7)
    plt.xlabel(r"$T/(J/k_B)$")
    plt.ylabel(r"$C_V/k_B$")
    plt.savefig("fig/CV.pdf", format="pdf")
    plt.clf()

def Burnin():
    EvO = pa.cube(); EvO.load("data/Evolution_ordered.dat"); EvO = np.array(EvO)
    EvR = pa.cube(); EvR.load("data/Evolution_random.dat"); EvR = np.array(EvR)

    N = np.linspace(1, 10000, 100)
    plt.plot(N, EvO[0, 0, :], label=r"<$\epsilon$>, T = 1.0")
    plt.plot(N, EvO[0, 1, :], label=r"<$\epsilon$>, T = 2.4")
    plt.plot(N, EvO[1, 0, :], label=r"<|m|>, T = 1.0")
    plt.plot(N, EvO[1, 1, :], label=r"<|m|>, T = 2.4")
    plt.title("Ordered initial state")
    plt.xlabel("Number of cycles")
    plt.legend()
    plt.savefig("fig/Ordered.pdf", format="pdf")
    plt.clf()

    plt.plot(N, EvR[0, 0, :], label=r"<$\epsilon$>, T = 1.0")
    plt.plot(N, EvR[0, 1, :], label=r"<$\epsilon$>, T = 2.4")
    plt.plot(N, EvR[1, 0, :], label=r"<|m|>, T = 1.0")
    plt.plot(N, EvR[1, 1, :], label=r"<|m|>, T = 2.4")
    plt.title("Random initial state")
    plt.xlabel("Number of cycles")
    plt.legend()
    plt.savefig("fig/Random.pdf", format="pdf")
    plt.clf()

Burnin()
plot_results()
Cv_peaks()
Histogram()
