#include <armadillo>
#include <cmath>
#include <complex>
#include <assert.h>

#include "Packet.hpp"

using namespace std;

void Simulate(string filename, double T_, bool single, bool duo, bool triple, double wy_){
    /*
    Simulation function inteded to make running the goal simulations easier
    Only inputs are variables which change for the four simulations done,
    all others are set in this function
    Saves result to armadillo complex matrix with given filename
    */
    
    // Initializing system
    double dt = 2.5e-5;
    double h = 0.005; // System size found from h with constraints x, y in [0, 1]
    WavePacket Wave(dt, h);
    if (single || duo || triple){ // If any -> Set slit, else V = 0
        Wave.DoubleSlit(single, triple); // Setting box to chosen-slit configuration
    }

    // Placing wave packet at desired location
    double xc = 0.25; // Horizontal centre
    double yc = 0.5;  // Vertical centre
    double px = 200;  // Horizonal momentum
    double py = 0;    // Vertical momentum
    double wx = 0.05; // Horizontal width (std. dev.)
    double wy = wy_;  // Vertical width
    Wave.PlacePacket(xc, yc, px, py, wx, wy);

    double T = T_; // Total runtime
    int steps = (int)(T/dt); // Number of steps in time
    int M = (int)(1.0/h); // System size with x,y in [0, 1]
    arma::cx_mat U(pow(M-2, 2), steps+2);

    // Adding time axis as first column
    arma::cx_vec t = arma::regspace<arma::cx_vec>(0.0, dt, T);
    U.col(0).fill(0.0); // Pad rest with zeros
    U.col(0).rows(0, steps) = t;

    // Saving initial system
    U.col(1) = Wave.u;

    for (int i = 2; i < steps+2; i++){
        cout << "\rStep: " << i-1 << " of " << steps << flush; // Progress tracker
        Wave.step(); // Take step in time
        U.col(i) = Wave.u; // Save new system state
    }
    cout << endl; // End progress tracker

    // Save result to file
    U.save(filename);
}

int main(){
    // First sim: No potential, T = 0.008, sigma_y = 0.05
    Simulate("data/No_potential.dat", 0.008, false, false, false, 0.05);

    // Second sim: Double slit, T = 0.002, sigma_y = 0.20
    Simulate("data/Double_slit.dat", 0.002, false, true, false, 0.20);
    
    // Third sim: Single slit, T = 0.002, sigma_y = 0.20
    Simulate("data/Single_slit.dat", 0.002, true, false, false, 0.20);
    
    // Fourth sim: Triple slit, T = 0.002, sigma_y = 0.20
    Simulate("data/Triple_slit.dat", 0.002, false, false, true, 0.20);
}