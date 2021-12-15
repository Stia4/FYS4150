// The WavePacket class

#ifndef __Packet_hpp__ // include guard
#define __Packet_hpp__

#include <armadillo>
#include <complex>

class WavePacket{
    public:
        int M;     // System size (includes boundaries)
        double dt; // Time step size (unitless)
        double h;  // Spatial step size (unitless), dx = dy = h

        arma::cx_vec u; // Wave vector in box (no boundaries)
        arma::mat V;    // Potential in box (no boundaries)

        // Constructor
        WavePacket(double dt, double h);

        // Callable functions to setup potential field
        void DoubleSlit(bool single, bool triple); // Sets potential as single, double, or triple slit

        // Place a waveform at index (i, j)
        void PlacePacket(double xc, double yc, double px, double py, double wx, double wy);

        // Do step in time
        void step();

    private:
        std::complex<double> r;  // Comp. variable, r = idt/2h^2
        double v0 = 1e10;        // Large potential intended as wall (used in V)

        arma::sp_cx_mat A;       // Crank-Nicolson matrices
        arma::sp_cx_mat B;

        int index(int i, int j); // Index support/translation
        void make_AB();           // Make Crank-Nicolson matrices
};

#endif