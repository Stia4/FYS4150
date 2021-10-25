import numpy as np                             #arrays and math
import matplotlib.pyplot as plt                #plots
import pyarma as pa                            #for importing armadillo cubes from C++
from matplotlib.animation import FuncAnimation #animated plots
from matplotlib import cm                      #colormaps for plots

def load_data(filename): # Load cubes
    r = pa.cube()
    r.load(filename) #Each cube has dimensions: (t, [xyz], particle #)
    r = np.array(r) #numpy arrays are easier to handle
    return r

def XY_Z(r, title=None, dt=None, filename=None): # XY-plane and Z(t) plots
    fig, ax = plt.subplots(2, 1, figsize=(9, 18))
    colors = cm.jet(np.linspace(0, 1, r.shape[-1]))
    iterations = np.arange(r.shape[0]) #"time" axis
    for i in range(r.shape[-1]):
        ax[0].plot(r[:, 0, i], r[:, 1, i], color=colors[i])
        if dt:
            ax[1].plot(iterations*dt, r[:, 2, i], color=colors[i])
        else:
            ax[1].plot(iterations, r[:, 2, i], color=colors[i])

    ax[0].plot(0, 0, "ko")
    ax[0].axis('equal')
    ax[0].set_xlabel("x [µm]")
    ax[0].set_ylabel("y [µm]")
    if dt:
        ax[1].set_xlabel("t [µs]")
    else:
        ax[1].set_xlabel("iteration #")
    ax[1].set_ylabel("z [µm]")
    plt.suptitle(title)
    plt.tight_layout()
    if filename != None:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()

def ThreeD(r, title=None, animate=False, filename=None): # 3D plot/animation
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_zlabel("z [µm]")
    ax.set_title(title)
    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=20
    colors = cm.jet(np.linspace(0, 1, r.shape[-1])) #individual color for each particle

    if animate == False: #do not animate
        if len(r.shape) > 2:
            for i in range(r.shape[-1]):
                ax.plot(r[:, 0, i], r[:, 1, i], r[:, 2, i], color=colors[i])
        else:
            ax.plot(r[:, 0], r[:, 1], r[:, 2], color=colors[0])
        
        if filename != None:
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()

    else: #do animation
        skip_factor = max([1, 10 * int(len(r)/1e5)])
        r = r[::skip_factor] #speed up animations by only using every skip_factor-th step
        if r.shape[-1] == 1:
            skip_factor /= 10 #single particle is slower, needs longer tail

        ln = np.array([[plt.plot([], [], [], color=colors[i], marker="o")[0],
                        plt.plot([], [], [], color=colors[i],     ls="-")[0]] for i in range(r.shape[-1])]).flatten()

        def init(): #initialiser func for animator
            ax.set_xlim(np.nanmin(r[:, 0]), np.nanmax(r[:, 0]))
            ax.set_ylim(np.nanmin(r[:, 1]), np.nanmax(r[:, 1]))
            ax.set_zlim(np.nanmin(r[:, 2]), np.nanmax(r[:, 2]))
            return ln

        def update(i): #update func for animator
            #setting tail length
            l = len(r[:i])
            if l>1000/skip_factor:
                l = i-int(1000/skip_factor)
            else:
                l = 0
            
            for j in range(r.shape[-1]):
                xi = r[i, 0, j] #ball marker
                yi = r[i, 1, j]
                zi = r[i, 2, j]
                ln[2*j].set_data(xi, yi)
                ln[2*j].set_3d_properties(zi)
                
                x = r[l:i, 0, j] #tail
                y = r[l:i, 1, j]
                z = r[l:i, 2, j]
                ln[2*j+1].set_data(x, y)
                ln[2*j+1].set_3d_properties(z)
            
            return ln

        ani = FuncAnimation(fig, update, frames=r.shape[0], init_func=init, blit=True, interval=0.0001)
        if filename != None:
            print(filename, 1)
            ani.save(filename, fps=60)
            print(filename, 2)
            plt.clf()
        else:
            plt.show()

def PhaseSpace(Space1, Space2, title=None, xlabel=None, ylabel=None, filename=None): # Arbituary phase space plots
    plt.figure(figsize=(10, 9))
    if len(Space1.shape) == 3: #three dimensional -> more than one particle
        for i in range(Space1.shape[-1]):
            plt.plot(Space1[:, i], Space2[:, i])
    else: #not 3D -> only one particle
        plt.plot(Space1, Space2)
        
    plt.axis("equal")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if filename != None:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()

def analytical_r(t, v0, x0, z0, w0, wz): # Analytical solution
    #v0 is v_y0
    
    w_pls = (w0 + np.sqrt(w0**2 - 2*wz**2))/2
    w_min = (w0 - np.sqrt(w0**2 - 2*wz**2))/2
    A_pls =  (v0 + w_min*x0)/(w_min - w_pls)
    A_min = -(v0 + w_pls*x0)/(w_min - w_pls)
    
    f = A_pls*np.exp(-1j*w_pls*t) + A_min*np.exp(-1j*w_min*t)
    z = z0*np.cos(wz*t)
    r = zip(np.real(f), np.imag(f), z)

    return np.array(list(r))

def rel_err_plot(r_list, dt_list, v0, x0, z0, RK4=None, filename=None): # Plot relative errors for interval t in [0, 100] µs
    #NB: r_list is only for single particle positions, e.g shape (t, xyz)
    #    dt_list is timesteps as strings
    w0 = 1*9.65e1/48 #assuming base values
    wz = np.sqrt(2*1*9.65e8/(48*10**8))
    delta_max = np.zeros(len(r_list)) #also estimating error convergence rate
    plt.figure(figsize=(11, 9))
    for i in range(len(r_list)): #for each r in r_list
        t = np.linspace(0, 100, len(r_list[i])) #get time-list, assuming t_end = 100µs
        
        r_analytical = analytical_r(t, v0, x0, z0, w0, wz) #analytical x,y
        r_a_len = np.sqrt(r_analytical[:, 0]**2 + r_analytical[:, 1]**2 + r_analytical[:, 2]**2) #vector length r
        x_a, y_a, z_a = r_analytical[:, 0], r_analytical[:, 1], r_analytical[:, 2]

        ri = r_list[i]
        r_abs = np.zeros(ri.shape[0])
        r_rel = np.zeros(ri.shape[0])
        for j in range(ri.shape[0]): #avoiding memory errors by using longer for-method
            r_abs[j] = abs(np.sqrt((x_a[j] - ri[j, 0])**2 + (y_a[j] - ri[j, 1])**2 + (z_a[j] - ri[j, 2])**2))
            r_rel[j] = r_abs[j]/r_a_len[j]
        delta_max[i] = np.nanmax(abs(r_abs))

        plt.plot(t, r_rel, label=dt_list[i]) #using absolute values

    #plt.legend(frameon=False, loc=3, ncol=len(r_list), fontsize=20)
    plt.legend(frameon=False, fontsize=20)
    plt.xlabel("t [µs]")
    plt.ylabel(r"r$_{\rm err}$")
    if RK4 == True:
        plt.ylim(5e-7, 5e0)
        method = " using Runge-Kutta 4th order"
    elif RK4 == False:
        plt.ylim(5e-6, 5e7)
        method = " using forward Euler"
    else:
        method = ""
    plt.title(r"Relative error for $\log_{10}({\rm dt})\in$["+dt_list[0][-2:]+", "+dt_list[-1][-1]+"]"+method)
    plt.yscale("log")
    plt.tight_layout()
    if filename != None:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()

    dt_list = np.array(dt_list, dtype=float) #converting dt to floats
    r_err = 1/4 * np.sum([np.log10(delta_max[i+1]/delta_max[i])
                         /np.log10(  dt_list[i+1]/dt_list[i]) for i in range(len(r_list)-1)])
    print("Error convergence rate estimate"+method+":", r_err)
    if RK4 == False:
        r_err = 1/4 * np.sum([np.log10(delta_max[i+2]/delta_max[i+1]) #only 4 smaller dt
                             /np.log10(  dt_list[i+2]/dt_list[i+1]) for i in range(len(r_list)-2)])
        print("Error convergence rate estimate"+method+" lower:", r_err)
        r_err = 1/4 * np.sum([np.log10(delta_max[i+3]/delta_max[i+2]) #only 3 smaller dt
                             /np.log10(  dt_list[i+3]/dt_list[i+2]) for i in range(len(r_list)-3)])
        print("Error convergence rate estimate"+method+" lowest:", r_err)

def Replicate_Figures(animation=False):
    """
    Replicates all figures used in project, then saves them to folder 'fig'.
    Saving animations takes a long time, so an optional argument is included
    to skip these and only save/plot images.
    Also prints convergence rates to terminal.
    """
    ### Setting default values for plots
    plt.rc('savefig', format="pdf") # Save figures as pdf
    plt.rc('figure', figsize=(10, 9)) # Larger figure sizes
    plt.rc('font', size=22) # Larger text

    ### Single particle
    r = load_data("data/single.r.dat")
    XY_Z(r,"Movement of a single particle in a Penning trap", 1e-3, "fig/single_xy_z.pdf")
    ThreeD(r, title="Motion of a single particle in 3D", filename="fig/single_3D.pdf") #Extra: 3D plot + animation
    if animation:
        ThreeD(r, title="Animation of single particle in 3D", animate=True, filename="fig/single_3D.mp4")


    ### Two particles
    r = [load_data("data/duo_coulomb.r.dat"),
        load_data("data/duo_separate.r.dat")]
    v = [load_data("data/duo_coulomb.v.dat"),
        load_data("data/duo_separate.v.dat")]

    XY_Z(r[0], "Motion of two particles with Coulomb forces", 1e-3, "fig/duo_xy_z_coulomb.pdf")
    XY_Z(r[1], "Motion of two particles without Coulomb forces", 1e-3, "fig/duo_xy_z_no_coulomb.pdf")

    PhaseSpace(r[1][:, 0, :], v[1][:, 0, :], r"Phase space [x, $v_x$] without Coulomb forces", "x [µm]", r"$v_x$ [µm/µs]", "fig/duo_x_vx_no_coulomb.pdf")
    PhaseSpace(r[1][:, 1, :], v[1][:, 1, :], r"Phase space [y, $v_y$] without Coulomb forces", "y [µm]", r"$v_y$ [µm/µs]", "fig/duo_y_vy_no_coulomb.pdf")
    PhaseSpace(r[1][:, 2, :], v[1][:, 2, :], r"Phase space [z, $v_z$] without Coulomb forces", "z [µm]", r"$v_z$ [µm/µs]", "fig/duo_z_vz_no_coulomb.pdf")
    PhaseSpace(r[0][:, 0, :], v[0][:, 0, :], r"Phase space [x, $v_x$] with Coulomb forces", "x [µm]", r"$v_x$ [µm/µs]", "fig/duo_x_vx_coulomb.pdf")
    PhaseSpace(r[0][:, 1, :], v[0][:, 1, :], r"Phase space [y, $v_y$] with Coulomb forces", "y [µm]", r"$v_y$ [µm/µs]", "fig/duo_y_vy_coulomb.pdf")
    PhaseSpace(r[0][:, 2, :], v[0][:, 2, :], r"Phase space [z, $v_z$] with Coulomb forces", "z [µm]", r"$v_z$ [µm/µs]", "fig/duo_z_vz_coulomb.pdf")

    ThreeD(r[1], "Motion of two particles without Coulomb forces", filename="fig/duo_3D_no_coulomb.pdf")
    ThreeD(r[0], "Motion of two particles with Coulomb forces", filename="fig/duo_3D_coulomb.pdf")
    if animation:
        ThreeD(r[0], "Motion of two particles with Coulomb forces", animate=True, filename="fig/duo_3D_coulomb.mp4") #Extra: Animation!


    ### Many different stepsizes, w/error convergence rate
    dt_list = ["1e0", "1e-1", "1e-2", "1e-3", "1e-4"]
    r_list = [load_data("data/single_RK4_"+str(dt)+".r.dat") for dt in dt_list]
    rel_err_plot(r_list, dt_list, 1, 1, 1, RK4=True, filename="fig/steps_RK4.pdf")
    r_list = [load_data("data/single_Euler_"+str(dt)+".r.dat") for dt in dt_list]
    rel_err_plot(r_list, dt_list, 1, 1, 1, RK4=False, filename="fig/steps_Euler.pdf")


    ### Time dependent electric field
    remaining = pa.mat()
    remaining.load("data/f_wV_remaining.dat")
    remaining = np.array(remaining)
    w_V = np.linspace(0.2, 2.5, 115*2)
    plt.plot(w_V, remaining[0], label="f = 0.1")
    plt.plot(w_V, remaining[1], label="f = 0.4")
    plt.plot(w_V, remaining[2], label="f = 0.7")
    plt.xlabel(r"$\omega_V$ [MHz]")
    plt.ylabel("Number of particles")
    plt.title("Particles remaining in trap after 500 µs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/remaining.pdf")
    plt.clf()

    ### Closeup of w_V = [0.3, 0.5]
    remaining = pa.cube()
    remaining.load("data/f_wV_closeup.dat")
    remaining = np.array(remaining)
    w_V = np.linspace(0.3, 0.5, 100)
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    ax[0].plot(w_V, remaining[0, 0], label="f = 0.1")
    ax[0].plot(w_V, remaining[0, 1], label="f = 0.4")
    ax[0].plot(w_V, remaining[0, 2], label="f = 0.7")
    ax[1].plot(w_V, remaining[1, 0], label="f = 0.1")
    ax[1].plot(w_V, remaining[1, 1], label="f = 0.4")
    ax[1].plot(w_V, remaining[1, 2], label="f = 0.7")
    ax[0].set_xlabel(r"$\omega_V$ [MHz]")
    ax[0].set_ylabel("Number of particles")
    ax[0].set_title("Coulomb forces disabled")
    ax[0].set_ylim(0, 100)
    ax[1].set_xlabel(r"$\omega_V$ [MHz]")
    ax[1].set_ylabel("Number of particles")
    ax[1].set_title("Coulomb forces enabled")
    ax[1].set_ylim(0, 100)
    plt.suptitle("Particles remaining in trap after 500 µs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/remaining_closeup.pdf")
    plt.clf()

if __name__ == "__main__": #__name__ guard
    Replicate_Figures(animation=False)