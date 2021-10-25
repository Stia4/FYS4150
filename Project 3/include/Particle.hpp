// The Particle class

#ifndef __Particle_hpp__ // include guard
#define __Particle_hpp__

#include <armadillo>

class Particle{
    public:
        float q_;     // charge, int if defined in terms of e?
        float m_;     // mass, int if defined in terms of u?
        arma::vec r_; // position
        arma::vec v_; // velocity

        // Constructor
        Particle(float charge, float mass, arma::vec position, arma::vec velocity);
        Particle();
};

#endif