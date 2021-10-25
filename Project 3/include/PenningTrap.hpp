// The PenningTrap class

#ifndef __PenningTrap_hpp__ // include guard
#define __PenningTrap_hpp__

#include "Particle.hpp" // using Particle class in declarations

#include <vector>    // standard vectors
#include <armadillo> // better vectors++

class PenningTrap{
    public:
        double B0_;
        double V0_;
        double d_;
        std::vector<Particle> trapped_particles;
        bool interact_;
        bool time_dependance_;
        double f_;
        double w_V_;
        double t_;
        double trap_range_;

        // Constructor
        PenningTrap(double magnetic_field_strength,
                    double electric_potential,
                    double characteristic_dimension,
                    bool interactions,
                    double timed_potential_amplitude,
                    double timed_potential_angfreq,
                    double trap_range);
        
        // Add particle to system, overloaded with single or vector of particles
        void add_particle(Particle particle);
        void add_particle(std::vector<Particle> particles);

        // Evaluate electric field
        arma::vec E_field(arma::vec r);

        // Evaluate magnetic field
        arma::vec B_field(arma::vec r);

        // Evaluate force between two particles
        arma::vec Force_particle(arma::vec ri, arma::vec rj, double qi, double qj);

        // Sum of forces on each particle from fields
        arma::mat Total_force_fields();

        // Sum of forces on each particle other particles
        arma::mat Total_force_particles();

        // Sum of forces from both other particles and fields
        arma::mat Total_force();

        // Simulate system for given dt, stepcount
        void Simulate(double dt, int iterations, bool RK4, std::string filename);

        // Take a step in time using RK4 method
        void step_RK4(double dt);

        // Take a step in time using forward Euler method
        void step_Euler(double dt);

        // Count number of particles within trap
        int Remaining_Particles();
};

#endif