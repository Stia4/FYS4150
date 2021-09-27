#include <iostream>  // enable input/output
#include <armadillo> // better vectors/matrices
#include <cmath>     // math functions/constants
#include <assert.h>  // consistency checks
using namespace std;

arma::mat make_tridiagonal(int N, double signature[3]){
    /*
    Makes tridiagonal matrix NxN with values along diagonals defined
    by input signature.

    Ex. N=3, signature = {1, 2, 3}:
    A = [[2, 3, 0],
         [1, 2, 3],
         [0, 1, 2]]

    Inputs:
        int N
            Size of output tridiagonal matrix NxN, min N = 2
        double signature[3]
            Standard vector length 3 containing values for sub-, main, and
            super-diagonal respectivly
    Returns:
        arma::mat A
            Tridiagonal Armadillo matrix size NxN
    */
    assert(N >= 2); // throw error if trying to create smaller than 2x2 matrix

    arma::mat A = arma::mat(N, N).fill(0.0); // initializing matrix

    // setting first and last row manually
    A(  0,   0) = signature[1]; // b_1, A(y, x)
    A(  0,   1) = signature[2]; // c_1
    A(N-1, N-1) = signature[1]; // b_n
    A(N-1, N-2) = signature[0]; // a_n

    // setting all other values
    for (int i=1; i<N-1; i++){
        A(i, i-1) = signature[0]; // a_i+1 | (indexing diff,
        A(i, i  ) = signature[1]; // b_i+1 |  i.e i=3 -> b_4)
        A(i, i+1) = signature[2]; // c_i+1 |
    }

    return A;
}

arma::vec analytical_eigenvalues(arma::mat A){
    /*
    Given a symmetric tridiagonal matrix A NxN, returns a vector
    containing eigenvalues calculated using analytical expression.
    
    Inputs:
        arma::mat A
            Armadillo matrix size NxN, assumed tridiagonal and symmetric
    Returns:
        arma::vec eigenvals
            Armadillo vector length N containing eigenvalues of A
    */
    assert(A.is_square());

    int N = A.n_cols;  // number of elements
    double d = A(0,0); // main diagonal
    double a = A(0,1); // sub-/super-diagonal

    arma::vec eigenvals = arma::vec(N); // N eigenvals for NxN matrix
    for (int i = 0; i < N; i++){
        eigenvals(i) = d + 2*a*cos((i+1)*M_PI/(N+1)); // M_PI = pi from cmath
    }

    return eigenvals;
}

arma::mat analytical_eigenvectors(arma::mat A){
    /*
    Given a symmetric tridiagonal matrix A NxN, returns a matrix
    containing eigenvectors calculated using analytical expression.
    
    Inputs:
        arma::mat A
            Armadillo matrix size NxN, assumed tridiagonal and symmetric
    Returns:
        arma::mat eigenvectors
            Armadillo matrix size NxN containing eigenvectors of A as columns
    */
    assert(A.is_square());

    int N = A.n_cols;
    
    arma::mat eigenvectors = arma::mat(N, N);
    for (int j = 0; j < N; j++){
        for (int i = 0; i < N; i++){
            eigenvectors(j, i) = sin((j+1)*(i+1)*M_PI/(N+1));
        }
    }
    eigenvectors = arma::normalise(eigenvectors, 2, 0);

    return eigenvectors;
}

double max_offdiag_symmetric(const arma::mat& A, int& k, int& l){
    /*
    Finds and returns largest absolute value off-diagonal element in symmetric matrix.
    Loops over upper triangular area and updates values when larger element is found.

    Input:
        arma::mat A
            Assumed symmetric, size NxN armadillo matrix, min 2x2
        int k, l
            Integers which will be updated with largest element index
    Updates:
        int k, l
            Set to indexes such that A(k, l) == largest_element
    Returns:
        double largest_element
            Value of largest absolute value off-diagonal element in matrix
    */
    assert(A.is_square() && A.n_cols >= 2); // require A size NxN and N >= 2
    
    double largest_element = 0.0; // double to store largest value
    int N = A.n_cols;             // assumed A size NxN

    for (int j = 0; j < N-1; j++) {     // ignoring diagonal/final row -> j_max = N-2
        for (int i = j+1; i < N; i++) { // from after diagonal i=j to rightmost N-1
            if (abs(A(j, i)) > abs(largest_element)) { // if a larger value is found
                largest_element = A(j, i);             // store found value
                k = j;                                 // and update indexes
                l = i;
            }
            // else continue;
        }
    }

    return largest_element;
}

void jacobi_rotate(arma::mat& A, arma::mat& R, int k, int l){
    /*
    Performs a single step using Jacobi's rotation method.
    
    Inputs:
        arma::mat A
            Symmetric Armadillo matrix defining eigenvalue problem, size NxN, N >= 2
        arma::mat R
            Symmetric Armadillo matrix storing rotations done
        int k, l
            Integers defining index of element to eliminate
    Updates:
        arma::mat A, R
            Gets updated with effect of rotation
    */
    // assert statements handled by parent function jacobi_eigensolver

    int N = A.n_cols;            // size of system
    double a_kk_temp, a_ik_temp; // temp variables to use after A is overwritten
    double r_ik_temp;            // similar for R

    // finding tau/sin/cos theta
    double tau = (A(l, l) - A(k, k))/(2 * A(k, l)); // computational variable
    double t; // tan(theta), solution to second degree equation w/coefficient tau
    if (tau > 0){ // choosing solution for smallest |t|, i.e smallest |theta|
        t =  1.0/( tau + sqrt( 1+pow(tau, 2) ));
    } else {
        t = -1.0/(-tau + sqrt( 1+pow(tau, 2) ));
    }
    double c = 1/sqrt(1+pow(t, 2)); // cos(theta), theta = rotation angle
    double s = c*t;                 // sin(theta)

    // performing rotation for A
    a_kk_temp = A(k, k); // used to update A(l, l)
    A(k, k) = A(k, k)*pow(c, 2) - 2*A(k, l)*c*s + A(l, l)*pow(s, 2); // setting kk, ll manually
    A(l, l) = A(l, l)*pow(c, 2) + 2*A(k, l)*c*s + a_kk_temp*pow(s, 2);
    A(k, l) = 0; // algorithm is based on these becoming zero
    A(l, k) = 0;
    for (int i = 0; i < N; i++){         // looping over other elements
        if (i == k || i == l){continue;} // skip cases i = k or l
        a_ik_temp = A(i, k);             // used to calculate il
        A(i, k) = A(i, k)*c - A(i, l)*s; // updating new using old
        A(i, l) = A(i, l)*c + a_ik_temp*s;
        A(k, i) = A(i, k);               // symmetrical A
        A(l, i) = A(i, l);
    }

    // updating R with same rotation
    for (int i = 0; i < N; i++){
        r_ik_temp = R(i, k); // temp variable for use in R(i, l)
        R(i, k) = R(i, k)*c - R(i, l)*s;
        R(i, l) = R(i, l)*c + r_ik_temp*s;
    }
}

void jacobi_eigensolver(arma::mat& A, arma::vec& eigenvalues, arma::mat& eigenvectors,
                        double eps, const int maxiter, int& iterations, bool& converged){
    /*
    Solves an eigenvalue system Av = λv using Jacobi's rotation method. Runs either
    until maximum off-diagonal element is smaller than eps, or until maximum number of
    iterations maxiter is reached.

    Inputs:
        arma::mat A
            Symmetric Armadillo matrix defining eigenvalue problem, size NxN, N >= 2
        arma::vec eigenvalues
            Armadillo vector length N which gets updated with eigenvalues of A
        arma::mat eigenvector
            Armadillo matrix size NxN which gets columns updated with eigenvectors
        double eps
            Treshold for when off-diagonal elements are considered small enough
        int maxiter
            Treshold for maximum number of iterations to compute, in case eps not crossed
        int iterations
            Count for number of iterations computed, updated for later reference
        bool converged
            Truth value for if any threshold eps, maxiter were crossed
    Updates:
        arma::vec eigenvalues
            Sets vector elements to eigenvalues of A
        arma::mat eigenvectors
            Sets columns of matrix to eigenvectors of A
        int iterations
            Updates with number of iterations computed during loop
        bool converged
            Once either threshold eps or maxiter are crossed, gets updated to true
    */
    // check assertions
    assert(A.is_square()                     // A size NxN
        && A.n_cols >= 2);                   // N >= 2
    assert(eigenvalues.n_elem == A.n_cols);  // eigenvalues length N
    assert(eigenvectors.is_square()          // eigenvectors square
        && eigenvectors.n_cols == A.n_cols); // eigenvectors 'sidelength' N
    
    // initialize variables
    int N = A.n_cols;
    int k, l;                                       // initializing indexes k,l
    max_offdiag_symmetric(A, k, l);                 // finding first k, l
    arma::mat R = arma::mat(N, N, arma::fill::eye); // initializing R as identity matrix

    // perform loop until any threshold is met
    while (abs(A(k, l)) > eps && maxiter > iterations){
        jacobi_rotate(A, R, k, l);      // perform rotation
        max_offdiag_symmetric(A, k, l); // find index of new largest element
        iterations++;                   // increment iteration count
    }
    if (abs(A(k, l)) < eps){
        converged = true; // only return true if eps treshold is met, not maxiter
    } else {
        converged = false;
    }

    // set eigenvalues and eigenvectors
    eigenvalues = A.diag(0);                 // converged A has eigenvalues on main diagonal
    eigenvectors = arma::normalise(R, 2, 0); // normalized eigenvectors, R ~ S
}

int main(){

    // Change booleans below to run code for each problem
    bool Problem3 = true;
    bool Problem4 = true;
    bool Problem5 = true;
    bool Problem6 = true;
    bool Problem7 = true;

    /* PROBLEM 3: Test eigenvalues */
    if (Problem3){
        int N = 6;
        double signature[3] = {1, 2, 1}; // simple symmetric signature
        arma::mat A = make_tridiagonal(N, signature);
        arma::vec eigenvalues = arma::vec(N);
        arma::mat eigenvectors = arma::mat(N, N);

        arma::eig_sym(eigenvalues, eigenvectors, A);

        cout << eigenvalues << endl;
        cout << analytical_eigenvalues(A) << endl;

        cout << eigenvectors << endl;
        cout << analytical_eigenvectors(A) << endl;
    }

    /* PROBLEM 4: Test off-diagonal*/
    if (Problem4){
        int N = 4;
        double signature[3] = {0, 1, 0}; // setting main diag to 1, rest 0
        arma::mat A = make_tridiagonal(N, signature);

        A(0, 3) =  0.5; // setting other elements
        A(1, 2) = -0.7;
        A(2, 1) = -0.7;
        A(3, 0) =  0.5;
        cout << A << endl;
        
        int k, l;
        double largest_element = max_offdiag_symmetric(A, k, l);
        cout << "Largest element: " << largest_element << endl;
        cout << "Index k: " << k << endl;
        cout << "Index l: " << l << endl;
    }

    /* PROBLEM 5: Test Jacobi */
    if (Problem5){
        int N = 6;
        double signature[3] = {1, 2, 1}; // simple signature for testing
        arma::mat A = make_tridiagonal(N, signature); // A matrix for testing
        cout << "A_initial =" << endl;
        cout << A << endl;
        arma::vec eigenvalues = arma::vec(N);         // vector for eigenvalues
        arma::mat eigenvectors = arma::mat(N, N);     // matrix for eigenvectors
        
        double eps = 1e-8;      // threshold value for off-diagonal size
        int maxiter = 100;      // threshold for maximum iteration number
        int iterations = 0;     // number of iterations computed
        bool converged = false; // if answer has converged

        jacobi_eigensolver(A, eigenvalues, eigenvectors, eps, maxiter, iterations, converged);

        if (!converged){
            int k, l;
            cout << "Result did not converge in " << maxiter << " iterations." << endl;
            cout << "Iterations done: " << iterations << endl;
            cout << "Eps = " << eps << endl;
            cout << "Max off-diag = " << max_offdiag_symmetric(A, k, l) << endl;
            cout << "k, l = " << k << ", " << l << endl;
            cout << A << endl;
        }
        else{
            cout << "Yay! Result converged in " << iterations << " iterations." << endl;
            cout << "Eps = " << eps << endl;
            cout << "A_final =\n" << A << endl;
            cout << "Eigenvalues =\n" << eigenvalues << endl;
            cout << "Analytical eigenvalues =\n" << analytical_eigenvalues(make_tridiagonal(N, signature)) << endl;
            cout << "Armadillo eigenvalues =\n" << arma::eig_sym(make_tridiagonal(N, signature)) << endl;
            cout << "Eigenvectors =\n" << eigenvectors << endl;
            arma::eig_sym(eigenvalues, eigenvectors, make_tridiagonal(N, signature));
            cout << "Armadillo eigenvectors =\n" << eigenvectors << endl;
            cout << "Analytical eigenvectors =\n" << analytical_eigenvectors(make_tridiagonal(N, signature)) << endl;
        }

    }

    /* PROBLEM 6: Scaling */
    if (Problem6){
        arma::vec N = arma::vec({  5,  10,  20,  30,  40,  50,  60,  70,  80,  90,
                                 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200});
        arma::vec iter_list(N.n_rows); // array to save iterations
        double eps = 1e-8;
        int maxiter = 100000;

        for (int i = 0; i < N.n_rows; i++){
            int iterations = 0;
            bool converged = false;
            double h2 = pow(1.0/(N(i)-1), 2);           // step size squared
            double signature[3] = {-1/h2, 2/h2, -1/h2}; // real problem signature
            arma::mat A = make_tridiagonal(N(i), signature);
            arma::vec eigenvalues = arma::vec(N(i));
            arma::mat eigenvectors = arma::mat(N(i), N(i));
            
            jacobi_eigensolver(A, eigenvalues, eigenvectors, eps, maxiter, iterations, converged);
            iter_list(i) = iterations; // storing iterations for current N

            if (converged){
                cout << "Matrix size N = " << N(i) << ", Iterations needed:" << iterations << endl;
            } else {
                cout << "Matrix size N = " << N(i) << ", Did not converge with " << iterations << " iterations" << endl;
            }
        }
        N.save("N.dat");                  // saving N-values for plotting in python
        iter_list.save("iterations.dat"); // saving iterations
        cout << "N and iteration count stored to files." << endl;
    }

    /* PROBLEM 7: Plotting */
    if (Problem7){
        int N = 10+1;                               // n steps -> N=n+1 points
        double h2 = pow(1.0/(N-1), 2);              // step size squared, n=N-1 steps in [0, 1]
        double signature[3] = {-1/h2, 2/h2, -1/h2}; // real problem signature
        arma::mat A = make_tridiagonal(N, signature);
        arma::vec eigenvalues = arma::vec(N);
        arma::mat eigenvectors = arma::mat(N, N);

        int iterations = 0;
        int maxiter = 1000;
        bool converged = false;
        double eps = 1e-8;

        jacobi_eigensolver(A, eigenvalues, eigenvectors, eps, maxiter, iterations, converged);

        arma::uvec indices = arma::sort_index(eigenvalues); // indices of sorted eigenvalues
        indices = indices.rows(0, 2);                       // picking 3 smallest
        arma::vec three_eigenvalues = eigenvalues(indices);
        arma::mat three_eigenvectors = eigenvectors.cols(indices);

        three_eigenvalues.save("eigenvalues.dat");
        three_eigenvectors.save("eigenvectors.dat");
        analytical_eigenvectors(make_tridiagonal(N, signature)).save("analytical_eigenvectors.dat");
        cout << "Eigenvalues and vectors saved to files." << endl;
    }

    return 0;
}
