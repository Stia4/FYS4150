// Definitions for the functions in the Particle class

#include "Particle.hpp"
#include <armadillo> // better vectors++

// Constructor
Particle::Particle(float charge, float mass, arma::vec position, arma::vec velocity){
    /*
    Constructor for class Particle. Sets input values as parameters of instance.

    Inputs:
        float charge
            Charge of particle in units of elementary charge [e]
        float mass
            Mass of particle in units of atomic mass [u]
        arma::vec position
            Armadillo vector of xyz-position, units in micrometers [µm]
        arma::vec velocity
            Armadillo vector of xyz-velocity, units [µm µs-1]
    */
    q_ = charge;
    m_ = mass;
    r_ = position;
    v_ = velocity;
}

Particle::Particle(){
    /* Overloading dummy particle for no specified values */
    q_ = 0;
    m_ = 0;
    r_ = arma::vec({0, 0, 0});
    v_ = arma::vec({0, 0, 0});
}