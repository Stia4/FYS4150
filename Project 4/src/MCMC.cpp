// Definitions for the functions in the MCMC (Markov Chain Monte Carlo) class

#include "MCMC.hpp"
#include "IsingModel.hpp"
#include <armadillo>
#include <cmath>
#include <random>
#include "omp.h"

// Costructor
MCMC::MCMC(int seed, int L_, arma::vec T_, double J_){
    /*
    desc
    */
    base_seed = seed; // Base used for other seeding

    L = L_; // System size
    T = T_; // Temperature array
    J = J_; // Coupling constant

    N = L*L; // Number of elements
}

arma::mat MCMC::Simulate(int N_Walks, int N_Burnin, bool ordered, bool silent){
    /*
    Performs N_Walks walks with N steps each,
    walks split across threads
    Only values after N_Burnin cycles are used
    */
    arma::mat Results(T.n_elem, 6); // Stores mean e, e2, m, m2 and Cv, X for each T

    // start thread split here
    #pragma omp parallel
    {
        int thread_number = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        if (thread_number == 0 && !silent){
            std::cout << "Beginning simulation for L = " << L << ", number of threads: " << num_threads << std::endl;
        }
        std::mt19937 gen(base_seed + thread_number); // Mersenne Twister number generator
        std::uniform_real_distribution<double> dist(0, 1); // [0, 1) distribution converter

        arma::Mat<int> Walk_Results(N_Walks, 4); // Sum over E, |M|, E^2, M^2 for each walk
        
        // looping over temperatures
        // start for threading, split temperatures across threads
        #pragma omp for
        for (int i = 0; i < T.n_elem; i++){
            if (thread_number == 0 && !silent){
                std::cout << "\rCurrent Step: " << i+1 << " of " << T.n_elem << "/" << num_threads << std::flush;
            }

            // Setting up model for T(i)
            IsingModel Model(L, T(i), J); // generating model
            if (!ordered){
                Model.Randomize(base_seed + i*T.n_elem + thread_number); // random initial state
            }

            // Begin walks
            for (int j = 0; j < N_Walks; j++){

                // Do walk/steps
                for (int k = 0; k < N; k++){

                    // Propose state and check acceptance
                    int idx_i = floor(dist(gen)*L);
                    int idx_j = floor(dist(gen)*L);
                    if (dist(gen) < Model.P(idx_i, idx_j)){ // min(1, P) part of P
                        Model.Flip(idx_i, idx_j); // if accepted -> do flip
                    }

                } // end steps

                Walk_Results(j, 0) =     Model.E    ; // E
                Walk_Results(j, 1) = pow(Model.E, 2); // E^2
                Walk_Results(j, 2) = abs(Model.M)   ; // |M|
                Walk_Results(j, 3) = pow(Model.M, 2); // M^2

            } // end walks

        // Mean values

        // Turns out arma::accu doesn't always work for some reason,
        // somehow gets negative values for sums over E^2
        // -> Using for loops instead
        Results(i, 0) = 0;
        Results(i, 1) = 0;
        Results(i, 2) = 0;
        Results(i, 3) = 0;
        for (int f = N_Burnin; f < N_Walks; f++){
            Results(i, 0) += Walk_Results(f, 0);
            Results(i, 1) += Walk_Results(f, 1);
            Results(i, 2) += Walk_Results(f, 2);
            Results(i, 3) += Walk_Results(f, 3);
        }

        } // end temperatures
    } // end parallel threads

    // Normalization and per spin <E> -> <e>
    for (int t = 0; t < T.n_elem; t++){
        Results(t, 4) = (Results(t, 1)/(N_Walks-N_Burnin) - pow(Results(t, 0)/(N_Walks-N_Burnin), 2))/(N*pow(T(t), 2)); // <e^2> - <e>^2
        Results(t, 5) = (Results(t, 3)/(N_Walks-N_Burnin) - pow(Results(t, 2)/(N_Walks-N_Burnin), 2))/(N*T(t));           // <m^2> - <|m|>^2

        Results(t, 0) /=   N*(N_Walks-N_Burnin); // N*(N_Walks-N_Burnin)
        Results(t, 1) /= N*N*(N_Walks-N_Burnin); // squares get an extra N
        Results(t, 2) /=   N*(N_Walks-N_Burnin);
        Results(t, 3) /= N*N*(N_Walks-N_Burnin);

    }

    return Results;
}

arma::rowvec MCMC::Solution_2x2(double J_, double T_){
    double epb8J = exp( 8 * J_/T_);              // exponentials appear often
    double emb8J = exp(-8 * J_/T_);

    double Z = 2*(emb8J + epb8J + 6);            // partition function

    double e  = 4*J_/Z * (emb8J - epb8J);        // mean E   per spin  [J]
    double e2 = 8*J_/Z * (emb8J + epb8J);        // mean E^2 per spin [J^2]
    double m  = 2/Z * (epb8J + 2);               // mean M   per spin  [1]
    double m2 = 2/Z * (epb8J + 1);               // mean M^2 per spin  [1]

    double Cv = 4/pow(T_, 2) * (e2 - pow(e, 2)); // heat capacity   [kB]
    double X  = 4/T_ * (m2 - pow(m, 2));         // susceptibility [J^-1]

    arma::rowvec Solution = arma::rowvec({{e}, {e2}, {m}, {m2}, {Cv}, {X}});

    return Solution;
}

arma::mat MCMC::Histogram(int N_Walks, int N_Burnin){
    
    arma::mat Occurances(T.n_elem, N, arma::fill::zeros);
    int extremal = 2*N;
    
    std::mt19937 gen(base_seed); // Mersenne Twister number generator
    std::uniform_real_distribution<double> dist(0, 1); // [0, 1) distribution converter    
    for (int i = 0; i < T.n_elem; i++){
        IsingModel Model(L, T(i), J);
        Model.Randomize(base_seed + i*T.n_elem);
        for (int j = 0; j < N_Walks; j++){
            for (int k = 0; k < N; k++){
                int idx_i = floor(dist(gen)*L);
                int idx_j = floor(dist(gen)*L);
                if (dist(gen) < Model.P(idx_i, idx_j)){
                    Model.Flip(idx_i, idx_j);
                }
                if (j >= N_Burnin){
                    Occurances(i, (Model.E + extremal)/4)++;
                }
            }
        }
    }

    return Occurances;
}