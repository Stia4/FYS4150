import numpy as np
from numpy.lib.npyio import load
import pyarma as pa
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

plt.rc('legend', frameon=False)
plt.rc('font', size=18)

def load_data(filename):
    """
    Load a complex matrix saved in C++ with armadillo,
    and return system size, wave function, and probability.
    Wave function and probability have been reshaped into matrices.
    """
    u = pa.cx_mat(); u.load(filename); u = np.array(u)
    M = int((len(u))**0.5 + 2)          # Get system size
    u = u.T                             # Swap time and position axis
    t = u[0].real                       # First vector is time steps
    u = u[1:]                           # Rest is system states
    u = u.reshape(u.shape[0], M-2, M-2) # Change from vector to matrix
    p = (u.conj()*u).real               # Get probability
    return M, t, u, p

def animate(filename, ThreeD=False, levels=10, save=None):
    M, t, _, p = load_data(filename)
    fig = plt.figure(figsize=(12, 12))
    if ThreeD:
        ax = plt.axes(projection="3d")
    else:
        ax = plt.axes()
        plt.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"t = {:.5f}, 1-$\Sigma$p = {:.3e}".format(t[0], 1-np.nansum(p[0])))

    #first frame
    X = np.linspace(0, 1, M-2)
    Y = np.linspace(0, 1, M-2)
    X, Y = np.meshgrid(X, Y)
    if ThreeD:
        ax.plot_surface(X, Y, p[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    else:
        plot = ax.contourf(X, Y, p[0], levels=levels)
        #plot = ax.imshow(p[0])
        #vmax = np.nanmax(p[0])
        fig.colorbar(plot)
        levels = plot.levels
        
    def data_gen(i): # janky update function
        ax.clear()
        if ThreeD:
            plot = ax.plot_surface(X, Y, p[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        else:
            plot = ax.contourf(X, Y, p[i], levels=levels)
            #plot = ax.imshow(p[i], vmax=vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"t = {:.5f}, 1-$\Sigma$p = {:.3e}".format(t[i], 1-np.nansum(p[i])))
        return plot

    ani = FuncAnimation(fig, data_gen, frames=p.shape[0], interval=30, blit=False)
    if save != None:
        ani.save(save, fps=30)
        plt.clf()
    else:
        plt.show()

def plot_frame(filename, t0=0, levels=10, save=None):
    M, t, _, p = load_data(filename)
    step = np.argmin(abs(t-t0))

    X = np.linspace(0, 1, M-2)
    Y = np.linspace(0, 1, M-2)
    X, Y = np.meshgrid(X, Y)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    img = ax.imshow(p[step], extent=(0, 1, 0, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"t = {:.5f}, 1-$\Sigma$p = {:.3e}".format(t[step], 1-np.nansum(p[step])))
    fig.colorbar(img)
    ax.set_aspect('equal', 'box')

    if save != None:
        plt.savefig(save, format="pdf")
        plt.clf()
    else:
        plt.show()

def detector_screen(filename, t0, x0=0.8, save=None):
    M, t, _, p = load_data(filename)
    tstep = np.argmin(abs(t-t0))
    xstep = np.argmin(abs(np.arange(0, 1, 1/M)-x0)) #x-x0

    X = np.linspace(0, 1, M-2)
    Y = np.linspace(0, 1, M-2)
    X, Y = np.meshgrid(X, Y)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(np.linspace(0, 1, p.shape[2]), p[tstep, :, xstep]/np.nansum(p[tstep, :, xstep]))
    ax.set_xlabel("y")
    ax.set_ylabel("p")

    if save != None:
        plt.savefig(save, format="pdf")
        plt.clf()
    else:
        plt.show()


### Creating plots
files = ["No_potential", "Double_slit",
         "Single_slit", "Triple_slit"]

for file in files:
    filename = "data/"+file+".dat"
    savename = "fig/"+file
    animate(filename, False, 100, save=savename+".mp4")
    plot_frame(filename, t0=0.000, levels=100, save=savename+"_snapshot_t0e-3.pdf")
    plot_frame(filename, t0=0.001, levels=100, save=savename+"_snapshot_t1e-3.pdf")
    plot_frame(filename, t0=0.002, levels=100, save=savename+"_snapshot_t2e-3.pdf")
    detector_screen(filename, t0=0.002, x0=0.8, save=savename+"_detector_x8e-1.pdf")

# Probability deviation for no slit case
plt.figure(figsize=(12, 12))
file = files[0]
filename = "data/"+file+".dat"
M,t,u,p = load_data(filename)
plt.plot([t[i] for i in range(p.shape[0])], [abs(1-np.nansum(p[step])) for step in range(p.shape[0])], label=file)
plt.xlabel("Time [1]")
plt.ylabel("Probability deviation")
plt.legend()
plt.savefig("fig/probability_none.pdf", format="pdf")
plt.clf()

# Probability deviation comparing for different number of slits
plt.figure(figsize=(12, 12))
for file in files[1:]:
    filename = "data/"+file+".dat"
    M,t,u,p = load_data(filename)
    plt.plot([t[i] for i in range(p.shape[0])], [abs(1-np.nansum(p[step])) for step in range(p.shape[0])], label=file)
plt.xlabel("Time [1]")
plt.ylabel("Probability deviation")
plt.legend()
plt.savefig("fig/probability_slits.pdf", format="pdf")
plt.clf()

t0 = 0.002
for file in files:
    filename = "data/"+file+".dat"
    savename = "fig/"+file
    M,t,u,p = load_data(filename)
    plt.figure(figsize=(12, 12))
    plt.imshow(u.real[np.where(t==t0)][0], extent=(0,1,0,1))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(savename+"_real.pdf", format="pdf")
    plt.clf()
    plt.imshow(u.imag[np.where(t==t0)][0], extent=(0,1,0,1))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(savename+"_imag.pdf", format="pdf")
    plt.clf()