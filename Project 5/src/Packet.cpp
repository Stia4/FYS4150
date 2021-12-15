// Definitions for the functions in the WavePacket class

#include "Packet.hpp"

#include <armadillo>
#include <complex>
#include <cmath>
#include <assert.h>

// Required to make complex numbers i write correctly,
// and 1.0j seems to be different type __complex__ double
// which does not play well with math functions (i.e. exp)
using namespace std::complex_literals; // Enable i as complex operator

WavePacket::WavePacket(double dt_, double h_){
    /*
    */
    dt = dt_;
    h = h_;
    M = (int)(1.0/h);
    r = 1.0i * dt/(2*h*h);

    // Initialize state vector to correct size
    u = arma::cx_vec(pow(M-2, 2));

    // Initializes by default with no potential
    V = arma::mat(M-2, M-2, arma::fill::zeros);
    make_AB();
}

int WavePacket::index(int i, int j){
    /*
    Given indices i,j, returns corresponding index k in
    column vector containing unwrapped system
    */
    return i + (M-2)*j;
}

void WavePacket::make_AB(){
    /*
    Generates matrices A, B used in matrix notation of Crank-Nicolson approach
    Intended as private function, hence takes no inputs.
    */

    // Sparse matrix for memory conservation, auto filled with zeros
    // Named A_, B_ as good practice to separate from class matrices A, B
    arma::sp_cx_mat A_(pow(M-2, 2), pow(M-2, 2));
    arma::sp_cx_mat B_(pow(M-2, 2), pow(M-2, 2));
    
    // Generating diagonal vector
    arma::cx_vec a(pow(M-2, 2));
    for (int i = 0; i < M-2; i++){
        for (int j = 0; j < M-2; j++){
            a(index(i, j)) = 1.0 + 4.0*r + 1.0i*dt/2.0 * V(i, j);
        }
    }

    // Filling matrix
    A_.diag() = a;
    A_.diag( 1).fill(-r); // Diagonals -r at +- 1
    A_.diag(-1).fill(-r);
    
    A_.diag(  M-2 ).fill(-r); // Diagonals -r at +- M-2
    A_.diag(-(M-2)).fill(-r);
    for (int k = M-2; k < A_.diag(1).n_elem; k += M-2){ // Setting 'holes'
        A_.diag( 1)(k) = 0.0;
        A_.diag(-1)(k) = 0.0;
    }

    A = A_;
    
    // A and B only differ in a few signs, so we can transform A into B
    B_ = A_ * -1;
    B_.diag(0) += 2.0;
    
    B = B_;
}

void WavePacket::DoubleSlit(bool single, bool triple){
    /*
    Makes a wall with one, two, or three slits in the potential
    By default makes two slits, but optional arguments enable one or three
    Rows and columns are swapped due to weird math, don't think about it too much
    */

    // Slit parameters (equal for all)
    double wx = 0.02; // Wall x-thickness
    double xc = 0.5;  // Wall x-position
    double ys = 0.05; // Slits inner separation
    double ya = 0.05; // Slit aperture

    arma::mat V_(M-2, M-2, arma::fill::zeros); // Making temporary matrix

    int col_min = floor((xc-wx)/h); // Round to wall columns
    int col_max = floor((xc+wx)/h);

    V_.rows(col_min, col_max).fill(v0); // Fill wall w/high potential

    if (single && !triple){
        int u_row = floor( (0.5 + ya/2)/h ); // Hole upper row index
        int l_row = floor( (0.5 - ya/2)/h ); // Lower row index

        V_.rows(col_min, col_max).cols(l_row, u_row).fill(0.0); // Making hole
    }

    else if (!single && !triple){
        int uu_row = floor( (0.5 + ys/2 + ya)/h ); // Upper hole upper row index
        int ul_row = floor( (0.5 + ys/2)/h );      // Upper hole lower row
        int lu_row = floor( (0.5 - ys/2)/h );      // Lower hole upper
        int ll_row = floor( (0.5 - ys/2 - ya)/h ); // etc.

        V_.rows(col_min, col_max).cols(ul_row, uu_row).fill(0.0); // Make holes
        V_.rows(col_min, col_max).cols(ll_row, lu_row).fill(0.0);
    }

    else {
        int uu_row = floor( (0.5 + 3*ya/2 + ys  )/h );
        int ul_row = floor( (0.5 +   ya/2 + ys  )/h );
        int cu_row = floor( (0.5          + ys/2)/h ); // Central upper
        int cl_row = floor( (0.5          - ys/2)/h );
        int lu_row = floor( (0.5 -   ya/2 - ys  )/h );
        int ll_row = floor( (0.5 - 3*ya/2 - ys  )/h );

        V_.rows(col_min, col_max).cols(ul_row, uu_row).fill(0.0);
        V_.rows(col_min, col_max).cols(cl_row, cu_row).fill(0.0);
        V_.rows(col_min, col_max).cols(ll_row, lu_row).fill(0.0);
    }

    V = V_;    // Store V in class
    make_AB(); // Remake A and B matrices with new V
}

void WavePacket::PlacePacket(double xc, double yc, double px, double py, double wx, double wy){
    /*
    Places a wave packet at given central indexes (ic, jc),
    with width/standard deviation (wx, wy) and momentum (px, py)
    */
    arma::cx_mat u_mat(M-2, M-2);

    // Create packet from formula
    double x, y;
    for (int i = 0; i < M-2; i++){
        for (int j = 0; j < M-2; j++){
            x = i*h;
            y = j*h;
            u_mat(i, j) = exp(-pow(x-xc, 2)/(2*wx*wx)
                              -pow(y-yc, 2)/(2*wy*wy)
                              +1.0i*px*(double)(x-xc) // force cast i-ic to double
                              +1.0i*py*(double)(y-yc)
                              );
        }
    }

    // Normalize packet such that probability is 1
    u_mat /= sqrt(arma::accu(arma::conj(u_mat)%u_mat)); // Note: % is element wise multiplication in armadillo

    // Add to total wave vector
    u = arma::vectorise(u_mat);
}

void WavePacket::step(){
    /*
    Takes single step in time by solving Au^(n+1) = Bu^n
    First, does matrix multiplication b = B*u^n,
    then solves Au^(n+1) = b using armadillo solver for sparse matrices
    */
    arma::cx_vec b = B*u;
    u = arma::spsolve(A, b);
}