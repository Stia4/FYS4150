// Definitions for the functions in the Particle class

#include "IsingModel.hpp"
#include <armadillo>
#include <cmath>

// Constructor
IsingModel::IsingModel(int L_, double T_, double J_){
    /*
    Sets up system with size LxL, initialized to all spin down
    Note: Input temperature is in units [J/kB] to alleviate calculations
    */
    L = L_;
    T = T_;
    J = J_;
    s = arma::Mat<int>(L, L).fill(-1);
    
    idx = arma::Col<int>(L+2); // Index system for periodic boundary
    for (int i = 0; i <= L+1; i++){
        idx(i) = (i-1+L)%L;
    }

    for (int i = 0; i < 5; i++){ // Pre-made probabilities for each possible dE
        dP[i] = std::min(1.0, exp(-dE[i]/T)); // dE defined in header
    }

    Update_Energy();        // Calculate initial energy
    Update_Magnetization(); // Calculate initial magnetization
}

// Overloaded empty constructor
IsingModel::IsingModel(){
    IsingModel(2, 1, 1);
}

// Randomize state
void IsingModel::Randomize(int seed){
    /*
    Randomizes state dependant on input seed
    Generates random integers 0 and 1, then transforms to -1 and 1
    */
    arma::arma_rng::set_seed(seed);
    s = arma::randi<arma::Mat<int>>(L, L, arma::distr_param(0, 1));
    s = s*2 - 1; // [0, 1] -> [-1, 1]

    Update_Energy();
    Update_Magnetization();
}

// Flip spin
void IsingModel::Flip(int i, int j){
    /*
    Finds dE for flip and updates, then flips, then
    finds change in magnetization
    */
    E += Energy_Change(i, j);
    
    s(i, j) *= -1;
    
    M += 2*s(i,j); // 1 -> -1: dM = -2, -1 -> 1: dM = 2
}

// Returns change in energy given a spin flip for position (i,j)
int IsingModel::Energy_Change(int i, int j){
    /*
    Must be run before actual flip
    */
    // boundary indexing is biased +1, i-1 -> i, i+1 -> i+2
    int sum = s(idx(i), j)
            + s(idx(i+2), j)
            + s(i, idx(j))
            + s(i, idx(j+2));
   
    // Possible sums = {-4, -2, 0, 2, 4}
    // -> Possible energies  = {-8, -4, 0,  4,  8}
    //    or if s(i,j) = -1 -> { 8,  4, 0, -4, -8}
    return 2 * sum * s(i, j);
}

// Calculate energy from current state
void IsingModel::Update_Energy(){
    /*
    Intended to be run only when initiating state,
    then energy is updated with Flip()
    E in units of J
    */
    E = -arma::accu(s % arma::shift(s, 1, 0)   // accu = sum over all elements
                  + s % arma::shift(s, 1, 1)); // % = element-wise * for arma
}

// Calculate magnetization from current state
void IsingModel::Update_Magnetization(){
    /*
    Intended to be run only when initiating state,
    then magnetization is updated with Flip()
    */
    M = arma::accu(s);
}

double IsingModel::P(int i, int j){
    /*
    Calculates relative probability between current state and proposed
    state given a spin flip for position (i,j)
    Given only 5 possible delta E, there are also only 5 possible P
    dE = {-8, -4, 0, 4, 8} -> (dE+8)/4 = {0, 1, 2, 3, 4}
    */
    return dP[(Energy_Change(i, j) + 8)/4];
}