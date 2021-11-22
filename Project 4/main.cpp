#include <iostream>
#include <armadillo>
#include <random>
#include <cmath>
#include <chrono>

#include "IsingModel.hpp"
#include "MCMC.hpp"

using namespace std;

arma::mat Simulate(int L, arma::vec T, double J, int seed, int N_Walks, int N_Burnin, string filename){
    MCMC Solver = MCMC(seed, L, T, J);
    arma::mat Results = Solver.Simulate(N_Walks, N_Burnin, false, false);
    if (filename != "nosave"){
        Results.save(filename);
    }
    return Results;
}

void Compare_2x2(){
    /*
    Runs simulation for 2x2 system and computes relative error
    compared to the analytical solution
    */
    // Checking error for different cycle counts
    int seed = 17051814;
    int N_Walks[5] = {10, 100, 1000, 10000, 100000};
    
    MCMC Solver = MCMC(seed, 2, arma::vec({1}), 1);
    arma::rowvec Solution = Solver.Solution_2x2(1, 1);
    
    arma::mat Results;
    for (int N : N_Walks){
        Results = Simulate(2, arma::vec({1}), 1, seed, N, 0, "nosave");
        cout << "\nCycles: " << N << ", Maximum relative error: " << max(abs((Solution-Results)/Solution)) << endl;
        cout << Solution << "\n" << Results << endl;
    }

    // Many values of T
    // Unused in project, only for debugging
    /*
    arma::vec T = arma::linspace(2.1, 2.4, 1000);
    double J = 1;
    Solver = MCMC(seed, 2, T, J);
    Results = Simulate(2, T, J, seed, 10000, 0, "Results_2x2.dat");
    arma::mat Rel_err(T.n_elem, 6);
    for (int i = 0; i < T.n_elem; i++){
        Solution = Solver.Solution_2x2(J, T(i));
        Rel_err.row(i) = abs((Solution - Results.row(i))/Solution);
    }
    Rel_err.save("Rel_err_2x2.dat");
    */
}

void Main_Simulations(){
    int seed = 17051814;
    arma::vec T = arma::linspace(2.1, 2.4, 100);
    double J = 1;
    int N_Walks = 500000;
    int N_Burnin = 3000;

    Simulate( 20, T, J, seed, N_Walks, N_Burnin, "data/Results_20x20_500k.dat");
    Simulate( 40, T, J, seed, N_Walks, N_Burnin, "data/Results_40x40_500k.dat");
    Simulate( 60, T, J, seed, N_Walks, N_Burnin, "data/Results_60x60_500k.dat");
    Simulate( 80, T, J, seed, N_Walks, N_Burnin, "data/Results_80x80_500k.dat");
    Simulate(100, T, J, seed, N_Walks, N_Burnin, "data/Results_100x100_500k.dat");
}

void Burn_in(){
    /*
    Could be implemented as a function of its own in MCMC class,
    but runtime is low enough that it's fine
    */
    int seed = 17051814;
    int L = 20;
    arma::vec T = arma::vec({1, 2.4});
    double J = 1;
    MCMC Solver = MCMC(seed, L, T, J);
    arma::vec N_Walks = arma::linspace(1, 10000, 100);
    arma::mat Results;
    arma::cube Evolution(2, N_Walks.n_elem, 2); // [T1/T2, N_Walks, e/m]

    // Ordered initial state
    for (int i = 0; i < N_Walks.n_elem; i++){
        Results = Solver.Simulate(N_Walks(i), 0, true, true);
        Evolution.slice(0).col(i) = Results.col(0);
        Evolution.slice(1).col(i) = Results.col(2);
        cout << "N = " << N_Walks(i) << endl;
    }
    Evolution.save("data/Evolution_ordered.dat");

    // Random initial state
    for (int i = 0; i < N_Walks.n_elem; i++){
        Results = Solver.Simulate(N_Walks(i), 0, false, true);
        Evolution.slice(0).col(i) = Results.col(0);
        Evolution.slice(1).col(i) = Results.col(2);
        cout << "N = " << N_Walks(i) << endl;
    }
    Evolution.save("data/Evolution_random.dat");
}

void Histogram(){
    int seed = 17051814;
    arma::vec T = arma::vec({1, 2.4});
    MCMC Solver(seed, 20, T, 1.0);
    arma::mat Occurances = Solver.Histogram(100000, 3000);
    Occurances.save("data/Histogram.dat");
}

void Time(){
    /* Relies on setting thread count manually from terminal
       export OMP_NUM_THREADS=1 */
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Simulate(20, arma::vec({1, 1.1, 1.2, 1.3}), 1, 17051814, 100000, 3000, "nosave");
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}

int main(){
    // Main programs have runtime of many many hours, so nothing is set to run as default
}