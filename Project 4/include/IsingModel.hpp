// The IsingModel class

#ifndef __IsingModel_hpp__ // include guard
#define __IsingModel_hpp__

#include <armadillo>

class IsingModel{
    public:
        // Inputs
        int L;              // System size
        double T;           // Temperature (units of J/kB)
        double J;           // Coupling constant

        // Calculated values
        arma::Mat<int> s;   // System state
        int E;              // State energy (units of J)
        int M;              // State magnetization

        // Constructor
        IsingModel(int L_, double T_, double J_);
        IsingModel();

        // Randomizes state dependant on input seed
        void Randomize(int seed);

        // Flip spin at index (i, j)
        void Flip(int i, int j);

        // Calculate relative probability between current state
        // and proposed given a spin flip at position (i,j)
        double P(int i, int j);

    private: // Variables/functions only intended for internal use
        // Computational variables
        int sum;        // used in dE calculation
        int E_proposed; // Energy for proposed spin flip

        // Pre-made possible spin state-changes to energy and probability
        int dE[5] = {-8, -4, 0, 4, 8};
        double dP[5]; // = exp(-dE/T)

        // Indexing/Boundary structure
        arma::Col<int> idx;

        // Get change in energy for proposed spin flip (i, j)
        int Energy_Change(int i, int j);

        // Re-calculate internal values
        void Update_Energy();
        void Update_Magnetization();
};

#endif