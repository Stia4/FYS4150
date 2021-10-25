#include <iostream>  // input/output enabler
#include <armadillo> // better vectors/matrices
#include <cmath>     // math functions
#include <vector>    // standard vector, for particle list
using namespace std; // avoid writing std:: everywhere

#include "PenningTrap.hpp" // class governing simulation
#include "Particle.hpp"    // class to contain particle values

PenningTrap Simulate(double B0, double V0, double f, double w_V, double d, double range,
                     bool coloumb, int N, bool random, double dt, int iterations,
                     bool use_RK4, string filename){
    /*
    Generates particles and sets up trap, then starts simulation using user input.
    Assumes Ca+ 48 isotopes as particle, and allows user to choose random or
    circular distribution of particles.

    Inputs:
        double B0
            Magnetic field strength, units [u µs-1 e-1]
        double V0
            Electric potential size, units [u µm2 µs-2 e-2]
        double f
            Amplitude factor of time dep. part of V0, 0: no time dependence
        double w_V
            Angular frequency of time dependent potential, units [µs-1]
        double d
            Characteristic dimension, units [µm]
        double range
            Range of electric/magnetic fields, 0: infinite, units [µm]
        bool coloumb
            Enable/disable Coloumb forces between particles
        int N
            Generated particle count
        bool random
            Randomize pos/vel, fixed seed, false: dist. as circle in xy + v=0, z=+-1
        double dt
            Timestep, unit [µs]
        int iterations
            Number of timesteps to simulate
        bool use_RK4
            Integration method, true: RK4, false: Euler
        std::string filename
            Prepend to file name, resulting name: filename+".{r/v/a}.dat"
            If filename="nosave", do not save to file (and do not print progress to terminal)
    Returns:
        PenningTrap PTrap
            Penning trap used for simulation
    */
    // Defining particles, set to Ca+ ions
    double q =  1; // Ca+ is once ionized
    double m = 48; // Ca+ isotope 48
    std::vector<Particle> particles; // empty list

    arma::vec r, v;
    if (random){ // if user wants random positions
        arma::arma_rng::set_seed(0);
        for (int i=0; i<N; i++){
            r = arma::vec(3).randn() * 0.1 * d;
            v = arma::vec(3).randn() * 0.1 * d;
            particles.push_back(Particle(q, m, r, v));
        }
    }    
    else { // if user wants even spaced positions
        if (N == 1){ // if N=1, start single particle at x0=1, v0=1, z0=1
            particles.push_back(Particle(q, m, arma::vec({1, 0, 1}), arma::vec({0, 1, 0})));}
        else {       // if N>1, distribute around circle
            arma::vec theta = arma::linspace(0, 2*3.14159265359, N+1); // spacing around a circle
            for (int i=0; i<N; i++){
                r = arma::vec({sin(theta(i)), cos(theta(i)), 0}); // turning angle to position
                v = arma::vec({0, 0, pow(-1, i)});                // every other has v_z = +- 1
                particles.push_back(Particle(q, m, r, v));
            }
        }
    }

    // Defining trap, and adding particles
    PenningTrap PTrap = PenningTrap(B0, V0, d, coloumb, f, w_V, range);
    PTrap.add_particle(particles);
    
    // Starting simulation
    PTrap.Simulate(dt, iterations, use_RK4, filename);

    return PTrap;
}

void Replicate_Project(){
    /*
    Runs all required simulations to recreate data used in project.
    Packed into a function outside main() to more easily be disabled in case other
    simulations should be run.
    Expected full runtime of ~10 hours, more or less depending on computer
    specification. Prints progress to terminal for convenience.
    */
    // Simulating single particle for 100 µs, see Simulate for input descriptions
    double B0 = 9.65e1; //  1  T in [u µs-1 e-1]
    double V0 = 9.65e8; // 10  V in [u µm2 µs-2 e-2]
    double d  = 1e4;    //  1 cm in [µm]
    Simulate(B0, V0, 0, 0, d, 0, true, 1, false, 1e-3, 100000, true, "data/single");
    
    // Two particles with and without interaction
    Simulate(B0, V0, 0, 0, d, 0, true, 2, false, 1e-3, 100000, true, "data/duo_coulomb");
    Simulate(B0, V0, 0, 0, d, 0, false, 2, false, 1e-3, 100000, true, "data/duo_separate");

    // Single particle for different stepsizes
    vector<string> dt_list   = {"1e0", "1e-1",  "1e-2",   "1e-3",    "1e-4"};
    vector<string> iter_list = {"100", "1000", "10000", "100000", "1000000"}; // all sum to 100µs
    string filename, dt, iter;
    for (int i=0; i<dt_list.size(); i++){
        dt = dt_list[i]; // easier to use string -> double/integer than reverse
        iter = iter_list[i];
        filename = "data/single_RK4_"+dt; // files are saved as "data/sim_{RK4/Euler}"+dt+".{r/v/a}.dat"
        Simulate(B0, V0, 0, 0, d, 0, true, 1, false, stod(dt), stoi(iter), true, filename);
        filename = "data/single_Euler_"+dt;
        Simulate(B0, V0, 0, 0, d, 0, true, 1, false, stod(dt), stoi(iter), false, filename);
    }

    // Reducing size of trap and using time-dependent E-field
    d  *= 0.05;   // reducing 1 cm to 0.05 cm
    V0 *= 2.5e-4; // reducing 10 V to 0.0025 V
    filename = "nosave"; // special string which skips saving
    dt = "1e-2";
    iter = "50000"; // dt * iter = 500 µs
    double range = d; // range of trap
    vector<double> f_list = {0.1, 0.4, 0.7}; // E-field swinging amplitude  
    arma::vec w_list = arma::linspace(0.2, 2.5, 115*2); // E-field angular frequency, dHz = 0.02 MHz -> 115 steps
    arma::mat num_left = arma::mat(f_list.size(), w_list.n_elem).fill(0.0); // to store particles left in trap
    PenningTrap PTrap(0,0,0,false,0,0,0); // "empty" instance to overwrite
    for (int i=0; i<f_list.size(); i++){
        for (int j=0; j<w_list.n_elem; j++){
            PTrap = Simulate(B0, V0, f_list[i], w_list(j), d, range, false, 100, true, stod(dt), stoi(iter), true, filename);
            num_left(i, j) = PTrap.Remaining_Particles(); // counting how many particles remain
            cout << "\rSimulations done: " << (i*w_list.n_elem)+j+1 << " of " << f_list.size()*w_list.n_elem << flush;
        }
    }
    num_left.save("data/f_wV_remaining.dat"); // saving to data file
    cout << "\nSimulation result saved to file 'data/f_wV_remaining.dat'" << endl;
    
    // Doing same experiment again, but for a narrower set of frequencies and Coulomb off/on
    w_list = arma::linspace(0.3, 0.5, 100); // Focusing on [0.3, 0.5], 100 steps
    arma::cube num_narrow = arma::cube(f_list.size(), w_list.n_elem, 2).fill(0.0);
    for (int i=0; i<f_list.size(); i++){
        for (int j=0; j<w_list.n_elem; j++){
            // without interactions
            PTrap = Simulate(B0, V0, f_list[i], w_list(j), d, range, false, 100, true, stod(dt), stoi(iter), true, filename);
            num_narrow(i, j, 0) = PTrap.Remaining_Particles();
            cout << "\rSimulations done: " << (2*i*w_list.n_elem)+2*j+1 << " of " << 2*f_list.size()*w_list.n_elem << flush;
            // with interactions
            PTrap = Simulate(B0, V0, f_list[i], w_list(j), d, range, true, 100, true, stod(dt), stoi(iter), true, filename);
            num_narrow(i, j, 1) = PTrap.Remaining_Particles();
            cout << "\rSimulations done: " << (2*i*w_list.n_elem)+2*j+2 << " of " << 2*f_list.size()*w_list.n_elem << flush;
        }
    }
    num_narrow.save("data/f_wV_closeup.dat"); // saving to data file
    cout << "\nSimulation result saved to file 'data/f_wV_closeup.dat'" << endl;
}

int main(){
    Replicate_Project();
}