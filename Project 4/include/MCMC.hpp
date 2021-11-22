// The MCMC (Markov Chain Monte Carlo) class

#ifndef __MCMC_hpp__ // include guard
#define __MCMC_hpp__

#include "IsingModel.hpp"
#include <armadillo>

class MCMC{
    public:
        // Inputs
        int base_seed;
        int L;
        arma::vec T;
        double J;

        // Calculated values
        double N; // Number of elements, L*L

        // Constructor
        MCMC(int seed, int L_, arma::vec T_, double J_);

        // Take walks
        arma::mat Simulate(int N_Walks, int N_Burnin, bool ordered, bool silent);

        // Analytical solution for L = 2
        arma::rowvec Solution_2x2(double J_, double T_);

        // Generate histogram of occurances
        arma::mat Histogram(int N_Walks, int N_Burnin);
};

#endif