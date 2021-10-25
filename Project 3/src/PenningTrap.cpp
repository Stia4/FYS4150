// Definitions for the functions in the PenningTrap class

#include "PenningTrap.hpp"
#include "Particle.hpp"

#include <vector>    // standard vectors
#include <armadillo> // better vectors++
#include <cmath>     // math functions

PenningTrap::PenningTrap(double magnetic_field_strength,
                         double electric_potential,
                         double characteristic_dimension,
                         bool interactions,
                         double timed_potential_amplitude,
                         double timed_potential_angfreq,
                         double trap_range){
    /*
    Constructor for class PenningTrap.
    Stores defining quantities of trap as member variables and
    initialises vector of stored particles.
    
    Units of inputs:
        magnetic_field_strength:   [u µs-1 e-1]
        electric_potential:        [u µm2 µs-2 e-1]
        characteristic_dimension:  [µm]
        timed_potential_amplitude: [1]
        timed_potential_angfreq:   [µs-1]
        trap_range:                [µm]
    
    Also stores if particles should interact in simulations,
    whether the potential is constant or time dependent
    (and constants related to this), and the range of the trap
    (with no limit if trap_range = 0).
    */
    B0_ = magnetic_field_strength;
    V0_ = electric_potential;
    d_  = characteristic_dimension;

    trapped_particles = std::vector<Particle> {};

    interact_ = interactions;
    f_ = timed_potential_amplitude;
    w_V_ = timed_potential_angfreq;
    t_ = 0.0; // initial time = 0
    trap_range_ = trap_range;
}

void PenningTrap::add_particle(Particle particle){
    /*
    Adds a given intance of Particle to system

    Inputs:
        Particle particle
            Instance of class Particle to add to system
    */
    
    trapped_particles.push_back(particle);
}

void PenningTrap::add_particle(std::vector<Particle> particles){
    /*
    Adds a given vector of Particle instances to system
    Overloaded class method to accept single or vector of Particle instances

    Inputs:
        std::vector<Particle> particles
            Vector of class Particle instances to add to system
    */
    for (Particle particle : particles){
        trapped_particles.push_back(particle);
    }
}

arma::vec PenningTrap::E_field(arma::vec r){
    /*
    Calculates the electric field at given position.
    
    Uses electric potential of form
        V(x,y,z) = V0/(2d^2) * (2z^2 - x^2 - y^2)
    which gives field
        E(x,y,z) = -∇V = V0/d^2 * [x, y, -2z]

    And V0 either constant or time dependent depending
    on f>0, with time dependent amplitude
        V0(t) = V0*(1+f*cos(w_V*t))
    with f, w_V set by constructor and t tracked by simulation.

    Inputs:
        arma::vec r
            Position for which to calculate field
    Returns:
        arma::vec E
            Resulting electric field for input position
    */
    arma::vec E;
    if (f_ > 0){ // Time dependant field
        double V0t = V0_ * (1 + f_*std::cos(w_V_ * t_));
        E = V0t/std::pow(d_, 2) * arma::vec({r(0), r(1), -2*r(2)});
    }
    else { // No time dependence
        E = V0_/std::pow(d_, 2) * arma::vec({r(0), r(1), -2*r(2)});
    }
    return E;
}

arma::vec PenningTrap::B_field(arma::vec r){
    /*
    Returns magnetic field at given position

    The magnetic field is simply defined as
        B(x,y,z) = [0, 0, B0]
    Meaning no actual position dependence

    Returns:
        arma::vec B
            Resulting magnetic field for input position
    */
    return arma::vec({0, 0, B0_}); // [0, 0, B0]
}

arma::vec PenningTrap::Force_particle(arma::vec ri, arma::vec rj, double qi, double qj){
    /*
    Calculates the Coulomb force on particle i from particle j,
    given their positions and charges.

    Force on particle i from particle j is calculated as:
        F_i = k_e*qi*qj * (ri-rj)/|ri-rj|^3

    Inputs:
        arma::vec ri,rj
            Vectors with the [x,y,z] positions of particles i,j
        double qi,qj
            Electric charge of particles i,j
    Returns:
        double F_i
            Coulomb force on particle i from particle j
    */
    arma::vec r = ri - rj;
    double r_len = std::sqrt(std::pow(r(0), 2) + std::pow(r(1), 2) + std::pow(r(2), 2));
    
    double k_e = 1.38935333e5; // coloumb constant [u µm3 µs-2 e2] 
    arma::vec F_i = k_e*qi*qj * r/std::pow(r_len, 3);

    return F_i;
}

arma::mat PenningTrap::Total_force_fields(){
    /*
    Calculates total force on each particle in system from the electric
    and magnetic fields.

    Uses force calculation:
        F_i = qi*E(ri) + qi*vi × B(ri)

    Returns:
        arma::mat field_forces
            Matrix with columns corresponding to list of particles, and
            rows x,y,z for force along each component
    */
    int N = trapped_particles.size();                   // number of particles
    arma::mat field_forces = arma::mat(3, N).fill(0.0); // matrix to store forces

    Particle pi; // empty particle
    for (int i=0; i<N; i++){
        pi = trapped_particles[i];
        if (arma::norm(pi.r_, 2) <= trap_range_ || trap_range_ == 0){
            field_forces.col(i) = pi.q_*(E_field(pi.r_) + arma::cross(pi.v_, B_field(pi.r_)));
        }
    }

    return field_forces;
}

arma::mat PenningTrap::Total_force_particles(){
    /*
    Calculates forces on each particle in system from other particles.
    Symmetry F_j = -F_i and j>i is used to limit number of calculations.

    Returns:
        arma::mat particle_forces
            Matrix with columns corresponding to list of particles, and
            rows x,y,z for force along each component
    */
    int N = trapped_particles.size();                      // number of particles
    arma::mat particle_forces = arma::mat(3, N).fill(0.0); // matrix to store forces

    Particle p1, p2; // particles to calculate forces between
    arma::vec F_i;   // force on i from j

    for (int i=0; i<N; i++){           // i =   0,   1, ..., N-2, N-1
        p1 = trapped_particles[i];     // particle i
        for (int j=i+1; j<N; j++){     // j = i+1, i+2, ..., N-2, N-1
            p2 = trapped_particles[j]; // particle j

            F_i = Force_particle(p1.r_, p2.r_, p1.q_, p2.q_);
            particle_forces.col(i) += F_i; // adding force to total
            particle_forces.col(j) -= F_i; // using symmetry
        }
    }

    return particle_forces;
}

arma::mat PenningTrap::Total_force(){
    /*
    Returns the total combined forces from Penning trap fields and other
    trapped particles on each particle in the system. Checks if system is
    set up with interacting particles or not.

    Returns:
        arma::mat total_forces
            Matrix with columns corresponding to list of particles, and
            rows x,y,z for force along each component
    */
    arma::mat F_fields = Total_force_fields();
    arma::mat F_particles = arma::mat(arma::size(F_fields)).fill(0.0);
    if (interact_){ // only calculate if interactions are enabled
        F_particles = Total_force_particles();}

    return F_fields + F_particles;
}

void PenningTrap::Simulate(double dt, int iterations, bool RK4, std::string filename){
    /*
    Simulates particle movements in the penning trap for given timestep and step count.
    Uses either Runge-Kutta 4th order or forward Euler integration method depending on
    user choice:
        RK4 ==  true: Uses Runge-Kutta method
        RK4 == false: Uses forward Euler method

    Results are stored to armadillo data-files, with exeption for input filename="nosave".

    Inputs:
        double dt
            Timestep of simulation, units [µs]
        int iterations
            Number of iterations to compute
        bool RK4
            Boolian which determines which integration method to use:
                true:  Runge-Kutta 4th order
                false: Forward Euler
        std::string filename
            String which gets prepended to filenames, final name: filename+".{r/v/a}.dat"
            Special behaviour for filename="nosave", which skips saving (and mutes printing)
    */
    int N = trapped_particles.size();                        // number of particles in system
    arma::cube pos = arma::cube(3, N, iterations).fill(0.0); // 3D vectors to store results
    arma::cube vel = arma::cube(3, N, iterations).fill(0.0);
    arma::cube acc = arma::cube(3, N, iterations).fill(0.0);

    void (PenningTrap::*step)(double);       // empty function step w/input double
    if (RK4){step = &PenningTrap::step_RK4;} // pointing to function depending on input
    else    {step = &PenningTrap::step_Euler;}
    
    for (int i=0; i<iterations; i++){
        (this->*step)(dt); // resolving pointer, taking step
        
        // Storing values to data files
        arma::mat F = Total_force(); // Calculating force twice, find better solution?
        for (int j=0; j<N; j++){
            pos.slice(i).col(j) = trapped_particles[j].r_;
            vel.slice(i).col(j) = trapped_particles[j].v_;
            acc.slice(i).col(j) = F.col(j)/trapped_particles[j].m_;
        }

        if (i % (iterations/100) == 0 && filename!= "nosave"){ // progress count
            std::cout << "\rProgress: " << 100*i/iterations+1 << " %" << std::flush;
        }

    }
    if (filename != "nosave"){
        pos.save(filename+".r.dat");
        vel.save(filename+".v.dat");
        acc.save(filename+".a.dat");
        std::cout << "\rSimulation saved to files '"+filename+".r.dat', '"+filename+".v.dat', '"+filename+".a.dat'" << std::endl;
    }
}

void PenningTrap::step_RK4(double dt){
    /*
    Step function for Runge-Kutta 4th order integration method.
    Takes timestep as input and updates PenningTrap instance's stored
    particles by one integration step.

    Having the forces calculated using internal trapped_particles leads
    to somewhat clunky implementation, as they have to be updated for each k.
    Could be improved in the future.

    Inputs:
        double dt
            Timestep to use for integration step, units [µs]
    */
    int N = trapped_particles.size(); // number of particles in system
    arma::mat pos0 = arma::mat(3, N);
    arma::mat vel0 = arma::mat(3, N);
    double t0 = t_; // also need to keep track of time for E_field
    for (int i=0; i<N; i++){pos0.col(i) = trapped_particles[i].r_;  // saving original positions
                            vel0.col(i) = trapped_particles[i].v_;} // saving original velocities
    
    // k1
    arma::mat k1_v = dt*Total_force();                               // calculating forces
    for (int i=0; i<N; i++){k1_v.col(i) /= trapped_particles[i].m_;} // a = F/m
    arma::mat k1_r = dt*(vel0);                                      // RHS of dx/dt is v(t)

    // k2
    t_ = t0 + dt/2;
    for (int i=0; i<N; i++){trapped_particles[i].r_ = pos0.col(i) + k1_r.col(i)/2;  // updating internal values
                            trapped_particles[i].v_ = vel0.col(i) + k1_v.col(i)/2;}
    arma::mat k2_v = dt*Total_force();                                              // get force w/new internal pos/vel
    for (int i=0; i<N; i++){k2_v.col(i) /= trapped_particles[i].m_;}
    arma::mat k2_r = dt*(vel0 + k1_v/2);                                            // v(ti+dt/2) = vi + k1/2
    
    // k3
    // t_ = t0 + dt/2; // same as before
    for (int i=0; i<N; i++){trapped_particles[i].r_ = pos0.col(i) + k2_r.col(i)/2;
                            trapped_particles[i].v_ = vel0.col(i) + k2_v.col(i)/2;}
    arma::mat k3_v = dt*Total_force();
    for (int i=0; i<N; i++){k3_v.col(i) /= trapped_particles[i].m_;}
    arma::mat k3_r = dt*(vel0 + k2_v/2);

    // k4
    t_ = t0 + dt; // updating time
    for (int i=0; i<N; i++){trapped_particles[i].r_ = pos0.col(i) + k3_r.col(i);
                            trapped_particles[i].v_ = vel0.col(i) + k3_v.col(i);}
    arma::mat k4_v = dt*Total_force();
    for (int i=0; i<N; i++){k4_v.col(i) /= trapped_particles[i].m_;}
    arma::mat k4_r = dt*(vel0 + k3_v);

    // actual step
    for (int i=0; i<N; i++){
        trapped_particles[i].r_ = pos0.col(i) + (  k1_r.col(i) + 2*k2_r.col(i)
                                              +  2*k3_r.col(i) +   k4_r.col(i))/6;
        trapped_particles[i].v_ = vel0.col(i) + (  k1_v.col(i) + 2*k2_v.col(i)
                                              +  2*k3_v.col(i) +   k4_v.col(i))/6;}
}

void PenningTrap::step_Euler(double dt){
    /*
    Step function for forward Euler integration method. Takes timestep as
    input and updates PenningTrap instance's stored particles by one
    integration step.

    Could potentially be vectorised/broadcast in future?

    Inputs:
        double dt
            Timestep to use for integration step, units [µs]
    */
    int N = trapped_particles.size(); // number of particles in system
    arma::mat F = Total_force();      // total force on each particle (fields + other)
    Particle pi;

    for (int i=0; i<N; i++){
        pi = trapped_particles[i];    // get particle
        
        pi.r_ += pi.v_*dt;            // updating position
        pi.v_ += (F.col(i)/pi.m_)*dt; // updating velocity

        trapped_particles[i] = pi;    // save particle
    }
    t_ += dt; // updating internal time (used for E_field)
}

int PenningTrap::Remaining_Particles(){
    /*
    Counts number of particles caught in trap. If trap is
    of infinite (zero) size, return number of total particles.
    
    Returns:
        int count
            Number of particles within trap range
    */
    if (trap_range_ == 0){ // if field is infinite
        return trapped_particles.size();
    }
    
    int count = 0;
    for (Particle pi : trapped_particles){
        if (arma::norm(pi.r_, 2) <= trap_range_){ // if |r| within limit, count it
            count++;
        }
    }
    return count;
}